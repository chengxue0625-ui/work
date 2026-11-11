# -*- coding: utf-8 -*-
"""
政府資料開放平台 CSV 抓取與視覺化分析 (Streamlit)

- 使用 requests.get() 抓取 CSV 網址（也可改上傳 CSV 檔）
- 自動判斷編碼 (UTF-8 / Big5)、自動把像數字的文字欄位轉成數字
- 只想抓資料：用「資料」「欄位概況」「下載」分頁即可
- 想分析：用「圖表分析」分頁，依欄位繪製長條圖、折線圖、圓餅圖、
  直方圖、散佈圖、箱型圖、旭日圖、樹狀圖、相關係數熱圖、地圖

執行方式:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import re
import warnings

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ======================================================================
# 設定與常數
# ======================================================================
DEFAULT_URL = "https://data.ntpc.gov.tw/api/datasets/781b822e-214a-4b9a-b4db-32c9f4626d98/csv/file"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
CHART_FONT = "Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif"

ENCODINGS = ["utf-8-sig", "cp950", "big5"]  # 自動偵測時依序嘗試
SEP_MAP = {"逗號 ,": ",", "分號 ;": ";", "Tab": "\t", "直線 |": "|"}

ID_HINTS = ["編號", "代碼", "代號", "電話", "郵遞區號", "統一編號", "身分證"]
DATE_HINTS = ["日期", "時間", "date", "time"]

AGGS = ["計數", "總和", "平均", "中位數", "最大值", "最小值"]
AGG_FUNC = {"總和": "sum", "平均": "mean", "中位數": "median", "最大值": "max", "最小值": "min"}

CHART_TYPES = ["長條圖", "折線圖", "圓餅圖", "直方圖", "散佈圖", "箱型圖",
               "旭日圖", "樹狀圖", "相關係數熱圖", "地圖"]
NONE_OPT = "(無)"
MAX_POINTS = 20000


# ======================================================================
# 抓取與解析
# ======================================================================
def fetch_csv(url: str, verify: bool, timeout: int) -> bytes:
    """用 requests.get() 抓取檔案內容 (bytes)。"""
    if not verify:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    resp = requests.get(url, headers=HEADERS, verify=verify, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def check_not_html(content: bytes) -> None:
    if not content or not content.strip():
        raise ValueError("下載到的內容是空的。")
    head = content[:500].lstrip().lower()
    if head.startswith(b"<!doctype html") or b"<html" in head:
        raise ValueError("這個網址回傳的是網頁 (HTML)，不是 CSV 檔。"
                         "請到資料集頁面複製『CSV』格式的下載連結。")


def parse_csv(content: bytes, enc_choice: str, sep_choice: str, skip_bad: bool, as_text: bool):
    encodings = ENCODINGS if enc_choice == "自動偵測" else [enc_choice]
    kwargs = {}
    if sep_choice == "自動偵測":
        kwargs.update(sep=None, engine="python")
    else:
        kwargs["sep"] = SEP_MAP[sep_choice]
    if skip_bad:
        kwargs["on_bad_lines"] = "skip"
    if as_text:
        # 先全部當文字讀入，避免 001 變成 1、電話開頭的 0 消失；
        # 之後由 convert_types() 判斷哪些欄位真的是數字
        kwargs["dtype"] = str

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, **kwargs)
            return df, enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"無法用 {'、'.join(encodings)} 解碼，請在側邊欄手動選擇其他編碼。")


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """把『像日期』『像數字』的文字欄位轉成正確型態；編號、電話、有前導 0 的欄位不轉。"""
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if not _is_text(s) or s.notna().sum() == 0:
            continue
        name_l = c.lower()

        # 1) 欄位名稱像日期/時間 → 嘗試轉 datetime
        if any(h in name_l for h in DATE_HINTS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() / s.notna().sum() >= 0.9:
                df[c] = dt
                continue

        # 2) 編號、代碼、電話等識別欄位，保持文字
        if any(h in c for h in ID_HINTS) or re.search(r"(^|[_\W])id($|[_\W])", name_l):
            continue

        # 3) 像數字 → 轉數值（去掉千分位逗號與 %）
        nn = (s.dropna().astype(str).str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False).str.strip())
        if nn.str.match(r"^0\d").any():  # 有前導 0（如 0912…）視為文字
            continue
        if pd.to_numeric(nn, errors="coerce").notna().mean() < 0.9:
            continue
        full = (s.astype(str).str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False).str.strip())
        num = pd.to_numeric(full.where(s.notna()), errors="coerce").astype("float64")
        if num.notna().all() and (num % 1 == 0).all():
            num = num.astype("int64")
        df[c] = num
    return df


@st.cache_data(show_spinner=False)
def load_df(content: bytes, enc_choice: str, sep_choice: str, skip_bad: bool,
            na_mode: str, convert: bool):
    check_not_html(content)
    df, enc = parse_csv(content, enc_choice, sep_choice, skip_bad, as_text=convert)
    df.columns = [str(c).strip() for c in df.columns]
    # 移除完全空白的 Unnamed 欄位
    drop = [c for c in df.columns if c.startswith("Unnamed") and df[c].isna().all()]
    df = df.drop(columns=drop)
    n_raw = len(df)
    if na_mode == "刪除含空值的列（任一欄）":
        df = df.dropna()
    elif na_mode == "刪除全部為空的列":
        df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    if convert:  # 先清掉空列再轉型，整數欄位才能保持整數
        df = convert_types(df)
    return df, enc, n_raw


# ======================================================================
# 圖表輔助函式
# ======================================================================
def classify(df: pd.DataFrame):
    num = [c for c in df.select_dtypes(include="number").columns]
    dt = [c for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns]
    cat = [c for c in df.columns if c not in num and c not in dt]
    return num, cat, dt


def uniq(name: str, existing) -> str:
    while name in existing:
        name += "_"
    return name


def aggregate(df, group_cols, value_col, agg):
    vname = uniq("統計值", group_cols)
    g = df.groupby(group_cols, dropna=False)
    if agg == "計數" or not value_col:
        out = g.size().reset_index(name=vname)
    else:
        out = g[value_col].agg(AGG_FUNC[agg]).reset_index(name=vname)
    return out, vname


def collapse_top(series: pd.Series, n: int) -> pd.Series:
    """只保留出現次數最多的前 n 類，其餘合併成『其他』；空值顯示為『(空值)』。"""
    s = series.astype(str).where(series.notna(), "(空值)")
    top = s.value_counts().head(n).index
    return s.where(s.isin(top), "其他")


def guess_col(cols, keys):
    for k in keys:
        for c in cols:
            if c.lower() == k or (not k.isascii() and k in c):
                return c
    return None


def default_cat_index(cols, cat_cols, df):
    for c in cat_cols:
        if 2 <= df[c].nunique() <= 50:
            return cols.index(c)
    return cols.index(cat_cols[0]) if cat_cols else 0


def sample_points(d: pd.DataFrame) -> pd.DataFrame:
    if len(d) > MAX_POINTS:
        st.caption(f"資料筆數較多，為了流暢度隨機抽樣 {MAX_POINTS:,} 筆繪圖。")
        return d.sample(MAX_POINTS, random_state=0)
    return d


def build_figure(chart, d, num_cols, cat_cols, dt_cols, all_cols):
    """依圖表類型顯示控制項並回傳 Plotly 圖；條件不足時回傳 None。"""
    other_cols = cat_cols + dt_cols  # 可當分類/分組使用的欄位

    # ------------------------------ 長條圖 ------------------------------
    if chart == "長條圖":
        x = st.selectbox("X 軸（分類欄位）", all_cols, index=default_cat_index(all_cols, cat_cols, d))
        color = st.selectbox("分組顏色（選填）", [NONE_OPT] + [c for c in other_cols if c != x])
        agg = st.selectbox("統計方式", AGGS)
        y = None
        if agg != "計數":
            if not num_cols:
                st.info("這份資料沒有數值欄位，只能使用「計數」。")
                return None
            y = st.selectbox("數值欄位", num_cols)
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider("最多顯示幾個類別", 5, 100, 20)
        sort_by = c2.radio("排序", ["依數值大到小", "依類別排序"], horizontal=True)
        horiz = c3.checkbox("橫向長條圖")

        group = [x] + ([color] if color != NONE_OPT else [])
        out, vname = aggregate(d, group, y, agg)
        totals = out.groupby(x, dropna=False)[vname].sum().sort_values(ascending=False)
        keep = totals.head(top_n).index
        out = out[out[x].isin(keep)]
        if sort_by == "依數值大到小":
            order = list(keep)
        else:
            try:
                order = sorted(keep)
            except TypeError:
                order = sorted(keep, key=str)
        order = [str(k) for k in order]
        out = out.assign(**{x: out[x].astype(str)})
        title = f"{x} 的{agg}" + (f"（{y}）" if y else "")
        if horiz:
            fig = px.bar(out, x=vname, y=x, color=None if color == NONE_OPT else color,
                         orientation="h", barmode="group", text_auto=True, title=title,
                         category_orders={x: order[::-1]})
        else:
            fig = px.bar(out, x=x, y=vname, color=None if color == NONE_OPT else color,
                         barmode="group", text_auto=True, title=title,
                         category_orders={x: order})
        return fig

    # ------------------------------ 折線圖 ------------------------------
    if chart == "折線圖":
        if not num_cols:
            st.info("折線圖需要至少一個數值欄位。")
            return None
        x_default = all_cols.index(dt_cols[0]) if dt_cols else 0
        x = st.selectbox("X 軸", all_cols, index=x_default)
        ys = st.multiselect("Y 軸（可多選數值欄位）", [c for c in num_cols if c != x],
                            default=[c for c in num_cols if c != x][:1])
        if not ys:
            st.info("請至少選擇一個 Y 軸欄位。")
            return None
        color = st.selectbox("分組顏色（選填）", [NONE_OPT] + [c for c in other_cols if c != x])
        agg = st.selectbox("同一個 X 值有多筆資料時", ["不彙總（逐筆連線）", "總和", "平均", "中位數"])

        keys = [x] + ([color] if color != NONE_OPT else [])
        work = d[keys + ys]
        if agg != "不彙總（逐筆連線）":
            work = work.groupby(keys, dropna=False)[ys].agg(AGG_FUNC[agg]).reset_index()
        work = work.sort_values(x)

        ind = uniq("指標", keys)
        val = uniq("數值", keys + [ind])
        long = work.melt(id_vars=keys, value_vars=ys, var_name=ind, value_name=val)
        multi = len(ys) > 1
        color_arg = color if color != NONE_OPT else (ind if multi else None)
        dash_arg = ind if (color != NONE_OPT and multi) else None
        fig = px.line(long, x=x, y=val, color=color_arg, line_dash=dash_arg,
                      markers=len(long) <= 300, title=f"{'、'.join(ys)}（依 {x}）")
        fig.update_yaxes(title="、".join(ys))
        return fig

    # ------------------------------ 圓餅圖 ------------------------------
    if chart == "圓餅圖":
        cat = st.selectbox("分類欄位", all_cols, index=default_cat_index(all_cols, cat_cols, d))
        agg = st.radio("統計方式", ["計數", "總和"], horizontal=True)
        y = None
        if agg == "總和":
            if not num_cols:
                st.info("這份資料沒有數值欄位，只能使用「計數」。")
                return None
            y = st.selectbox("數值欄位", num_cols)
        top_n = st.slider("最多顯示幾個類別（其餘合併為『其他』）", 3, 30, 10)

        out, vname = aggregate(d, [cat], y, agg)
        out[cat] = out[cat].astype(str)
        out = out[out[vname] > 0].sort_values(vname, ascending=False)
        if len(out) > top_n:
            rest = out.iloc[top_n:][vname].sum()
            out = pd.concat([out.head(top_n), pd.DataFrame({cat: ["其他"], vname: [rest]})],
                            ignore_index=True)
        fig = px.pie(out, names=cat, values=vname, hole=0.3,
                     title=f"{cat} 的{agg}佔比" + (f"（{y}）" if y else ""))
        fig.update_traces(textposition="inside", textinfo="percent+label", sort=False)
        return fig

    # ------------------------------ 直方圖 ------------------------------
    if chart == "直方圖":
        if not num_cols:
            st.info("直方圖需要至少一個數值欄位。")
            return None
        col = st.selectbox("數值欄位", num_cols)
        c1, c2 = st.columns(2)
        bins = c1.slider("分組數", 5, 100, 30)
        marginal = c2.checkbox("上方加箱型圖")
        color = st.selectbox("分組顏色（選填，類別數 ≤ 20）",
                             [NONE_OPT] + [c for c in cat_cols if d[c].nunique() <= 20])
        fig = px.histogram(d, x=col, nbins=bins, color=None if color == NONE_OPT else color,
                           marginal="box" if marginal else None, title=f"{col} 分布")
        return fig

    # ------------------------------ 散佈圖 ------------------------------
    if chart == "散佈圖":
        if len(num_cols) < 2:
            st.info("散佈圖需要至少兩個數值欄位。")
            return None
        c1, c2 = st.columns(2)
        x = c1.selectbox("X 軸", num_cols, index=0)
        y = c2.selectbox("Y 軸", num_cols, index=1)
        color = st.selectbox("顏色（選填）", [NONE_OPT] + [c for c in d.columns if c not in (x, y)])
        size = st.selectbox("點大小（選填）", [NONE_OPT] + [c for c in num_cols if c not in (x, y)])
        p = sample_points(d)
        if size != NONE_OPT:
            p = p[p[size].fillna(0) > 0]
        fig = px.scatter(p, x=x, y=y, color=None if color == NONE_OPT else color,
                         size=None if size == NONE_OPT else size,
                         hover_data=list(p.columns[:8]), title=f"{y} vs {x}")
        return fig

    # ------------------------------ 箱型圖 ------------------------------
    if chart == "箱型圖":
        if not num_cols:
            st.info("箱型圖需要至少一個數值欄位。")
            return None
        y = st.selectbox("數值欄位", num_cols)
        x = st.selectbox("分組欄位（選填）", [NONE_OPT] + cat_cols)
        p = d[[y] + ([x] if x != NONE_OPT else [])].copy()
        if x != NONE_OPT:
            top_n = st.slider("最多顯示幾組", 3, 30, 12)
            p[x] = collapse_top(p[x], top_n)
        fig = px.box(p, x=None if x == NONE_OPT else x, y=y, points="outliers",
                     title=f"{y} 分布" + (f"（依 {x}）" if x != NONE_OPT else ""))
        return fig

    # --------------------------- 旭日圖 / 樹狀圖 ---------------------------
    if chart in ("旭日圖", "樹狀圖"):
        default = [c for c in cat_cols if 2 <= d[c].nunique() <= 30][:2]
        path = st.multiselect("階層欄位（依選擇順序由外而內，1–4 個）", all_cols,
                              default=default, max_selections=4)
        if not path:
            st.info("請至少選擇一個階層欄位。")
            return None
        agg = st.radio("大小依據", ["計數", "總和"], horizontal=True)
        y = None
        if agg == "總和":
            if not num_cols:
                st.info("這份資料沒有數值欄位，只能使用「計數」。")
                return None
            y = st.selectbox("數值欄位", num_cols)
            if y in path:
                st.warning(f"「{y}」已經被選為階層欄位，請改選其他數值欄位或移除該階層。")
                return None
        top_n = st.slider("每一層最多保留幾個類別（其餘合併為『其他』）", 3, 30, 10)

        tmp = pd.DataFrame({c: collapse_top(d[c], top_n) for c in path})
        vname = uniq("數量", path)
        if y:
            tmp[y] = d[y]
            tmp = tmp[tmp[y] > 0]
            out = tmp.groupby(path)[y].sum().reset_index(name=vname)
        else:
            out = tmp.groupby(path).size().reset_index(name=vname)
        fn = px.sunburst if chart == "旭日圖" else px.treemap
        fig = fn(out, path=path, values=vname, title=f"{chart}：{' → '.join(path)}")
        fig.update_traces(textinfo="label+percent parent")
        return fig

    # ------------------------------ 熱圖 ------------------------------
    if chart == "相關係數熱圖":
        if len(num_cols) < 2:
            st.info("相關係數熱圖需要至少兩個數值欄位。")
            return None
        cols = st.multiselect("數值欄位", num_cols, default=num_cols[:12])
        method = st.radio("相關係數", ["pearson", "spearman"], horizontal=True)
        if len(cols) < 2:
            st.info("請至少選擇兩個欄位。")
            return None
        corr = d[cols].corr(method=method)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto", title=f"相關係數熱圖（{method}）")
        return fig

    # ------------------------------ 地圖 ------------------------------
    if chart == "地圖":
        if len(num_cols) < 2:
            st.info("地圖需要經度、緯度兩個數值欄位。")
            return None
        lat_g = guess_col(num_cols, ["緯度", "lat", "latitude", "wgsy", "y"])
        lon_g = guess_col(num_cols, ["經度", "lon", "lng", "longitude", "wgsx", "x"])
        c1, c2 = st.columns(2)
        lat = c1.selectbox("緯度欄位", num_cols, index=num_cols.index(lat_g) if lat_g else 0)
        lon = c2.selectbox("經度欄位", num_cols, index=num_cols.index(lon_g) if lon_g else min(1, len(num_cols) - 1))
        hover = st.selectbox("滑鼠移上去顯示的名稱（選填）", [NONE_OPT] + list(d.columns))
        color = st.selectbox("顏色（選填）", [NONE_OPT] + [c for c in d.columns if c not in (lat, lon)])
        p = d[(d[lat].between(-90, 90)) & (d[lon].between(-180, 180))]
        p = p[(p[lat] != 0) | (p[lon] != 0)]
        if p.empty:
            st.warning("找不到有效的經緯度（緯度需介於 -90~90、經度需介於 -180~180），"
                       "請確認是否選反了欄位。")
            return None
        p = sample_points(p)
        zoom = st.slider("縮放程度", 1, 16, 10)
        kwargs = dict(lat=lat, lon=lon, zoom=zoom, height=650,
                      hover_name=None if hover == NONE_OPT else hover,
                      color=None if color == NONE_OPT else color)
        scatter_map = getattr(px, "scatter_map", None)
        if scatter_map is not None:
            fig = scatter_map(p, **kwargs)
            fig.update_layout(map_style="open-street-map")
        else:
            fig = px.scatter_mapbox(p, **kwargs)
            fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig

    return None


# ======================================================================
# 畫面
# ======================================================================
def main():
    st.set_page_config(page_title="政府開放資料 CSV 分析", page_icon="🏛️", layout="wide")
    st.title("🏛️ 政府開放資料 CSV 抓取與分析")
    st.caption("輸入政府資料開放平台的 CSV 下載連結，抓取資料後可直接瀏覽、下載，或依欄位繪製圖表。")

    with st.sidebar:
        st.header("資料來源")
        url = st.text_input("CSV 網址", value=DEFAULT_URL,
                            help="到 data.gov.tw 或各縣市開放資料平台，找到資料集的『CSV』下載連結並貼上。")
        c1, c2 = st.columns(2)
        timeout = c1.number_input("逾時秒數", 5, 120, 30)
        skip_ssl = c2.checkbox("略過 SSL 驗證", value=False,
                               help="遇到憑證錯誤 (SSLError) 才勾選。只建議用在公開資料。")
        fetch_clicked = st.button("📥 抓取資料", type="primary")

        uploaded = st.file_uploader("或上傳 CSV 檔", type=["csv", "txt"])

        st.header("讀取與清理")
        enc_choice = st.selectbox("編碼", ["自動偵測", "utf-8-sig", "utf-8", "cp950", "big5"])
        sep_choice = st.selectbox("分隔符號", ["自動偵測"] + list(SEP_MAP.keys()))
        skip_bad = st.checkbox("略過格式錯誤的列", value=False)
        convert = st.checkbox("自動轉換數字／日期欄位", value=True,
                              help="編號、代碼、電話或有前導 0 的欄位會保持文字。")
        na_mode = st.selectbox("空值處理", ["不處理", "刪除全部為空的列", "刪除含空值的列（任一欄）"])

    # ---------- 取得內容 ----------
    if fetch_clicked:
        if not url.strip():
            st.error("請輸入 CSV 網址。")
        else:
            try:
                with st.spinner("正在抓取資料…"):
                    content = fetch_csv(url.strip(), verify=not skip_ssl, timeout=int(timeout))
                st.session_state["content"] = content
                st.session_state["source"] = url.strip()
            except requests.exceptions.SSLError as e:
                st.error(f"SSL 憑證驗證失敗：{e}\n\n可在側邊欄勾選「略過 SSL 驗證」後再試一次。")
            except requests.exceptions.Timeout:
                st.error("連線逾時，請稍後再試或調高逾時秒數。")
            except requests.exceptions.HTTPError as e:
                st.error(f"伺服器回應錯誤：{e}")
            except requests.exceptions.RequestException as e:
                st.error(f"抓取失敗：{e}")

    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if st.session_state.get("upload_key") != key:
            st.session_state["upload_key"] = key
            st.session_state["content"] = uploaded.getvalue()
            st.session_state["source"] = f"上傳檔案：{uploaded.name}"

    content = st.session_state.get("content")
    if content is None:
        st.info("👈 請在左側貼上 CSV 網址並按「抓取資料」，或直接上傳 CSV 檔。")
        st.stop()

    # ---------- 解析 ----------
    try:
        df, enc, n_raw = load_df(content, enc_choice, sep_choice, skip_bad, na_mode, convert)
    except Exception as e:
        st.error(f"CSV 解析失敗：{e}")
        st.stop()

    if df.empty:
        st.error("資料讀取成功，但清理後沒有任何資料列。請調整左側的「空值處理」。")
        st.stop()

    removed = n_raw - len(df)
    st.caption(f"來源：{st.session_state.get('source', '')}　｜　編碼：{enc}"
               + (f"　｜　已移除 {removed:,} 列空值資料" if removed > 0 else ""))

    num_cols, cat_cols, dt_cols = classify(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("資料列數", f"{len(df):,}")
    m2.metric("欄位數", f"{df.shape[1]:,}")
    m3.metric("數值欄位", f"{len(num_cols)}")
    m4.metric("缺漏值總數", f"{int(df.isna().sum().sum()):,}")

    # ---------- 篩選 ----------
    mask = pd.Series(True, index=df.index)
    with st.expander("🔧 篩選資料（選填，會同時套用到所有分頁）"):
        fcols = st.multiselect("選擇要篩選的欄位", list(df.columns))
        for c in fcols:
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                continue
            if pd.api.types.is_numeric_dtype(s):
                sv = s.dropna()
                if sv.empty or sv.min() == sv.max():
                    st.caption(f"「{c}」沒有可篩選的範圍。")
                    continue
                if pd.api.types.is_integer_dtype(s):
                    lo, hi = int(sv.min()), int(sv.max())
                else:
                    lo, hi = float(sv.min()), float(sv.max())
                sel = st.slider(c, lo, hi, (lo, hi))
                if sel != (lo, hi):
                    mask &= s.between(sel[0], sel[1])
            elif pd.api.types.is_datetime64_any_dtype(s):
                sv = s.dropna()
                if sv.empty or sv.min() == sv.max():
                    st.caption(f"「{c}」沒有可篩選的範圍。")
                    continue
                lo, hi = sv.min().to_pydatetime(), sv.max().to_pydatetime()
                sel = st.slider(c, lo, hi, (lo, hi))
                if sel != (lo, hi):
                    mask &= s.between(pd.Timestamp(sel[0]), pd.Timestamp(sel[1]))
            else:
                opts = s.dropna().astype(str).value_counts().index.tolist()[:1000]
                sel = st.multiselect(c, opts)
                if sel:
                    mask &= s.astype(str).isin(sel)

    fdf = df[mask]
    if len(fdf) != len(df):
        st.caption(f"篩選後剩 {len(fdf):,} 列。")
    if fdf.empty:
        st.warning("篩選後沒有資料，請放寬條件。")
        st.stop()

    f_num, f_cat, f_dt = classify(fdf)
    all_cols = list(fdf.columns)

    tab_data, tab_info, tab_chart, tab_dl = st.tabs(["📋 資料", "🔎 欄位概況", "📊 圖表分析", "⬇️ 下載"])

    # ---- 資料 ----
    with tab_data:
        show_cols = st.multiselect("顯示欄位", all_cols, default=all_cols)
        q = st.text_input("關鍵字搜尋（搜尋所有欄位）")
        view = fdf[show_cols] if show_cols else fdf
        if q.strip():
            hit = view.astype(str).apply(lambda col: col.str.contains(q.strip(), case=False, na=False, regex=False))
            view = view[hit.any(axis=1)]
            st.caption(f"符合「{q.strip()}」的資料：{len(view):,} 列")
        st.dataframe(view, height=500)

    # ---- 欄位概況 ----
    with tab_info:
        info = pd.DataFrame({
            "欄位": fdf.columns,
            "型態": [str(t) for t in fdf.dtypes],
            "非空值": fdf.notna().sum().values,
            "缺漏值": fdf.isna().sum().values,
            "缺漏率(%)": (fdf.isna().mean() * 100).round(1).values,
            "不重複值": fdf.nunique(dropna=True).values,
            "範例": [str(s.dropna().iloc[0]) if s.notna().any() else "" for _, s in fdf.items()],
        })
        st.dataframe(info, height=min(600, 40 + 35 * len(info)))
        if f_num:
            st.markdown("**數值欄位統計**")
            st.dataframe(fdf[f_num].describe().T)

    # ---- 圖表分析 ----
    with tab_chart:
        chart = st.selectbox("圖表類型", CHART_TYPES)
        fig = build_figure(chart, fdf, f_num, f_cat, f_dt, all_cols)
        if fig is not None:
            fig.update_layout(font_family=CHART_FONT)
            if chart != "地圖":
                fig.update_layout(height=600)
            st.plotly_chart(fig)

    # ---- 下載 ----
    with tab_dl:
        st.write("下載目前（套用清理與篩選後）的資料，或原始檔案。")
        st.download_button(
            "⬇️ 下載整理後的 CSV (UTF-8-SIG)",
            data=fdf.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="opendata_cleaned.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ 下載原始檔案",
            data=content,
            file_name="opendata_original.csv",
            mime="text/csv",
        )


main()
