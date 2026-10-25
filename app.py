import os
import math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ==========================================
# 1. 『実戦 化学重要問題集』マスター設定 (全275問 / 旧帝大=100,000pt)
# ==========================================
GOAL_POINTS = 100000
DATA_FILE = "jumon_progress_data.csv"

JUMON_CHAPTERS = [
    ("理論化学", "第1章 物質の構成粒子", 1, 14, 11, 4500),
    ("理論化学", "第2章 物質量と化学反応式", 15, 26, 9, 4000),
    ("理論化学", "第3章 化学結合と結晶", 27, 41, 11, 5000),
    ("理論化学", "第4章 物質の三態・気体の法則", 42, 58, 11, 6500),
    ("理論化学", "第5章 溶液の性質", 59, 77, 12, 7000),
    ("理論化学", "第6章 化学反応と熱・光（エンタルピー）", 78, 90, 9, 4500),
    ("理論化学", "第7章 反応の速さと化学平衡", 91, 108, 11, 7000),
    ("理論化学", "第8章 酸と塩基の反応・中和滴定", 109, 125, 11, 6000),
    ("理論化学", "第9章 酸化還元反応・電池・電気分解", 126, 148, 15, 7500),
    ("無機化学", "第10章 非金属元素", 149, 163, 11, 5500),
    ("無機化学", "第11章 典型金属元素", 164, 182, 13, 6500),
    ("無機化学", "第12章 遷移元素・錯イオン・系統分離", 183, 198, 10, 6000),
    ("有機化学", "第13章 脂肪族化合物・構造決定の基礎", 199, 218, 13, 7500),
    ("有機化学", "第14章 芳香族化合物・有機化合物の分離", 219, 237, 12, 7500),
    ("有機化学", "第15章 天然有機化合物（糖・アミノ酸・核酸）", 238, 253, 10, 6000),
    ("有機化学", "第16章 合成高分子化合物（繊維・樹脂・ゴム）", 254, 266, 8, 5000),
    ("有機化学", "巻末補充問題 旧帝大・難関大総合演習", 267, 275, 0, 4000),
]

STATUS_MULTIPLIER = {
    "未着手": 0.0,
    "解けなかった": 0.1,
    "一部解けた": 0.5,
    "解けた": 1.0,
}

STAGES = [
    ("基礎固め", "基礎固め I", 0, 10000),
    ("基礎固め", "基礎固め II", 10000, 22000),
    ("基礎固め", "基礎固め III", 22000, 35000),
    ("得点力養成", "得点力養成 I", 35000, 50000),
    ("得点力養成", "得点力養成 II", 50000, 65000),
    ("実戦力完成", "実戦力完成 I", 65000, 82000),
    ("実戦力完成", "実戦力完成 II", 82000, 95000),
    ("旧帝大突破", "旧帝大合格圏", 95000, 100000),
]

CAUSE_TAGS = [
    "知識・暗記抜け",
    "立式・方針迷い",
    "計算ミス",
    "条件見落とし",
    "時間不足",
    "完全理解・良問"
]


def build_master_dataframe() -> pd.DataFrame:
    rows = []
    for field, chapter, start_no, end_no, a_count, chap_goal in JUMON_CHAPTERS:
        total_q = end_no - start_no + 1
        b_count = total_q - a_count
        weight_sum = a_count * 1.0 + b_count * 1.45
        base_unit = chap_goal / weight_sum

        allocated = 0
        for idx, q_no in enumerate(range(start_no, end_no + 1)):
            if chapter.startswith("巻末補充"):
                q_type = "巻末補充"
                default_lvl = 9 if idx < 5 else 10
                weight = 1.45
            elif idx < a_count:
                q_type = "A問題"
                default_lvl = 4 if idx < (a_count // 2) else 6
                weight = 1.0
            else:
                q_type = "B問題"
                default_lvl = 8 if idx < total_q - 2 else 9
                weight = 1.45

            if q_no == end_no:
                max_pt = chap_goal - allocated
            else:
                max_pt = int(round(base_unit * weight))
                allocated += max_pt

            rows.append({
                "問題番号": q_no,
                "分野": field,
                "章": chapter,
                "区分": q_type,
                "レベル": default_lvl,
                "満点pt": max_pt,
                "解答状況": "未着手",
                "獲得pt": 0,
                "周回数": 0,
                "最終学習日": "-",
                "原因タグ": "",
                "つまずきメモ": "",
                "解法のコツ": ""
            })
    return pd.DataFrame(rows)


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_FILE):
        df_loaded = pd.read_csv(DATA_FILE)
        for col in ["原因タグ", "つまずきメモ", "解法のコツ"]:
            if col not in df_loaded.columns:
                df_loaded[col] = ""
            else:
                df_loaded[col] = df_loaded[col].fillna("").astype(str)
        return df_loaded
    df_init = build_master_dataframe()
    df_init.to_csv(DATA_FILE, index=False)
    return df_init


def save_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)


def evaluate_jumon_image(img: Image.Image, q_type: str, has_shikou_mark: bool) -> int:
    gray = np.array(img.convert("L"))
    contrast = float(np.std(gray))
    edges = (
        np.abs(np.diff(gray.astype(float), axis=0)).mean()
        + np.abs(np.diff(gray.astype(float), axis=1)).mean()
    )
    visual_complexity = (edges * 0.35) + (contrast * 0.04)
    base_offset = 0 if q_type == "A問題" else (2 if q_type == "B問題" else 3)
    shikou_bonus = 1 if has_shikou_mark else 0
    return int(np.clip(round(visual_complexity + base_offset + shikou_bonus), 1, 10))


def get_stage_info(points: int):
    for idx, (cat, name, low, high) in enumerate(STAGES):
        if points < high or idx == len(STAGES) - 1:
            next_name = STAGES[min(idx + 1, len(STAGES) - 1)][1]
            rem = max(0, high - points)
            span = max(1, high - low)
            stage_pct = min(100.0, max(0.0, (points - low) / span * 100.0))
            return cat, name, next_name, rem, stage_pct
    return "基礎固め", "基礎固め I", "基礎固め II", 10000, 0.0


def render_polar_chart_svg(chap_df: pd.DataFrame, target_ratio: float = 0.72) -> str:
    size = 360
    cx, cy = size / 2, size / 2
    max_r = 132
    n = len(chap_df)

    color_map = {
        "理論化学": ["#39D353", "#7CE38B", "#4AC26B", "#82E596", "#2DA44E", "#6BD983", "#56D364", "#8CE99A", "#3FB950"],
        "無機化学": ["#F7DF54", "#F5CD38", "#FFE875"],
        "有機化学": ["#549EF7", "#6FB0FA", "#85BEFC", "#448EF6", "#9ECBFF"]
    }

    svg = [f'<svg viewBox="0 0 {size} {size}" width="100%" style="max-width:380px;display:block;margin:0 auto;">']
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="#FFFFFF" stroke="#000000" stroke-width="1.5"/>')

    for frac in [0.2, 0.4, 0.6, 0.8]:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r * frac:.1f}" fill="none" stroke="#E2E0D8" stroke-width="1"/>')

    field_counters = {"理論化学": 0, "無機化学": 0, "有機化学": 0}
    for i, row in chap_df.iterrows():
        start_deg = -90 + (i * 360 / n)
        end_deg = -90 + ((i + 1) * 360 / n)
        a1 = math.radians(start_deg)
        a2 = math.radians(end_deg)

        gx = cx + max_r * math.cos(a1)
        gy = cy + max_r * math.sin(a1)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{gx:.1f}" y2="{gy:.1f}" stroke="#E2E0D8" stroke-width="1"/>')

        pct = min(100.0, max(0.0, float(row["進捗率(%)"])))
        r_val = max_r * (max(pct, 4.0) / 100.0) if pct > 0 else 0

        field = row["分野"]
        c_list = color_map.get(field, ["#549EF7"])
        fill_color = c_list[field_counters[field] % len(c_list)]
        field_counters[field] += 1

        if r_val > 0:
            x1 = cx + r_val * math.cos(a1)
            y1 = cy + r_val * math.sin(a1)
            x2 = cx + r_val * math.cos(a2)
            y2 = cy + r_val * math.sin(a2)
            path_d = f"M {cx},{cy} L {x1:.1f},{y1:.1f} A {r_val:.1f},{r_val:.1f} 0 0,1 {x2:.1f},{y2:.1f} Z"
            svg.append(f'<path d="{path_d}" fill="{fill_color}" fill-opacity="0.88" stroke="#FFFFFF" stroke-width="0.6"/>')

        mid_rad = math.radians((start_deg + end_deg) / 2)
        lx = cx + (max_r + 16) * math.cos(mid_rad)
        ly = cy + (max_r + 16) * math.sin(mid_rad) + 4
        svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="800" fill="#000000" font-family="sans-serif">{i + 1}</text>'
        )

    target_r = max_r * target_ratio
    svg.append(
        f'<circle cx="{cx}" cy="{cy}" r="{target_r:.1f}" fill="none" '
        f'stroke="#C93B3B" stroke-width="2" stroke-dasharray="5,4"/>'
    )
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{max_r}" fill="none" stroke="#000000" stroke-width="1.8"/>')
    svg.append("</svg>")
    return "".join(svg)


def stars_html(level: int) -> str:
    filled = int(np.clip(math.ceil(level / 2), 1, 5))
    s = ""
    for i in range(5):
        if i < filled:
            s += '<span style="color:#52D669; -webkit-text-stroke: 1px #000000; font-size:16px;">★</span>'
        else:
            s += '<span style="color:#FFFFFF; -webkit-text-stroke: 1px #000000; font-size:16px;">★</span>'
    return s


# ==========================================
# 2. ページ設定 ＆ カスタムCSS
# ==========================================
st.set_page_config(page_title="SIRUS - 化学重要問題集", page_icon="🏛️", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #EEECE2;
    color: #000000;
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", sans-serif;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 560px !important;
}
.hl-yellow {
    background: linear-gradient(transparent 55%, #F7DF54 55%);
    font-weight: 800;
    padding: 0 3px;
}
.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    border-bottom: 1px solid #DCD9CA;
    margin-bottom: 12px;
}
.brand-logo {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #000000;
    display: flex;
    align-items: center;
    gap: 8px;
}
.icon-pill {
    border: 1.5px solid #000000;
    border-radius: 999px;
    padding: 4px 14px;
    background: #F5F4EC;
    font-size: 16px;
    display: flex;
    gap: 12px;
}
.univ-header {
    font-size: 20px;
    font-weight: 800;
    color: #000000;
    margin: 6px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stage-bar {
    display: flex;
    gap: 4px;
    margin-bottom: 14px;
}
.stage-step {
    flex: 1;
    text-align: center;
    padding: 7px 4px;
    font-size: 12px;
    font-weight: 700;
    background: #DFDDD0;
    color: #000000;
    clip-path: polygon(0% 0%, 90% 0%, 100% 50%, 90% 100%, 0% 100%, 10% 50%);
}
.stage-step:first-child {
    clip-path: polygon(0% 0%, 90% 0%, 100% 50%, 90% 100%, 0% 100%);
    border-radius: 4px 0 0 4px;
}
.stage-step.active {
    background: #000000;
    color: #FFFFFF;
}
.pill-progress-outer {
    width: 100%;
    height: 30px;
    background: #FFFFFF;
    border: 1.8px solid #000000;
    border-radius: 999px;
    padding: 3px;
    box-sizing: border-box;
    margin: 10px 0 14px 0;
}
.pill-progress-inner {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #E8F54A 0%, #38C958 100%);
    transition: width 0.4s ease;
}
.dotted-box {
    background-color: #F8F7F2;
    background-image: radial-gradient(#D5D2C4 1.2px, transparent 1.2px);
    background-size: 12px 12px;
    border-top: 1.5px solid #000000;
    border-bottom: 1.5px solid #000000;
    padding: 16px 14px;
    margin: 18px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stack-wrapper {
    position: relative;
    width: 175px;
    height: 150px;
    margin: 10px auto;
}
.stack-card {
    position: absolute;
    width: 145px;
    height: 130px;
    background: #FFFFFF;
    border: 1.5px solid #000000;
    border-radius: 4px;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.12);
    padding: 8px;
    font-size: 11px;
}
.stack-tab {
    position: absolute;
    top: -18px;
    left: 4px;
    background: #FFFFFF;
    border: 1.5px solid #000000;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 1px 8px;
    font-size: 10px;
    font-weight: 700;
    color: #000000;
}
.date-badge {
    position: absolute;
    right: -18px;
    background: #65A2F8;
    color: #000000;
    border: 1.2px solid #000000;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 800;
}
.q-cards-row {
    display: flex;
    gap: 8px;
    justify-content: space-between;
    margin: 14px 0 18px 0;
}
.q-card {
    flex: 1;
    background: #FFFFFF;
    border: 1.8px solid #000000;
    box-shadow: 3px 4px 0px #000000;
    padding: 10px 8px 12px 8px;
    position: relative;
    min-height: 175px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.q-card.tilt-left { transform: rotate(-2.5deg); }
.q-card.tilt-mid { transform: rotate(0.8deg); margin-top: 8px; }
.q-card.tilt-right { transform: rotate(2.5deg); }
.pt-circle {
    position: absolute;
    top: -8px;
    right: -6px;
    background: #7CE38B;
    border: 1.5px solid #000000;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 900;
    line-height: 1.05;
    color: #000000;
}
.solve-pill {
    align-self: flex-end;
    background: #000000;
    color: #FFFFFF;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
}
.burst-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #F7DF54;
    border: 1.5px solid #000000;
    border-radius: 50%;
    width: 34px;
    height: 34px;
    font-weight: 900;
    font-size: 14px;
    margin-right: 8px;
    color: #000000;
}
.memo-card {
    background: #FFFFFF;
    border: 1.8px solid #000000;
    box-shadow: 3px 4px 0px #000000;
    border-radius: 6px;
    padding: 14px;
    margin-bottom: 14px;
    position: relative;
    color: #000000;
}
.memo-tag-chip {
    display: inline-block;
    background: #F7DF54;
    border: 1.2px solid #000000;
    border-radius: 999px;
    padding: 1px 9px;
    font-size: 11px;
    font-weight: 800;
    margin-right: 5px;
    margin-bottom: 4px;
    color: #000000;
}
.memo-box-trouble {
    background: #FFF5F5;
    border-left: 4px solid #E05252;
    padding: 8px 10px;
    margin-top: 8px;
    font-size: 13px;
    border-radius: 0 4px 4px 0;
}
.memo-box-tip {
    background: #F2FBF4;
    border-left: 4px solid #38C958;
    padding: 8px 10px;
    margin-top: 6px;
    font-size: 13px;
    border-radius: 0 4px 4px 0;
}
div[data-testid="stTabs"] button {
    font-weight: 800 !important;
    font-size: 13px !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. データ集計と状態管理
# ==========================================
df = load_data()

if "selected_q" not in st.session_state:
    st.session_state.selected_q = 1
if "target_univ" not in st.session_state:
    st.session_state.target_univ = "九州大学 理系(非医学部)"

total_earned = int(df["獲得pt"].sum())
total_pct = round((total_earned / GOAL_POINTS) * 100, 2)
solved_count = int((df["解答状況"] == "解けた").sum())
answered_count = int((df["解答状況"] != "未着手").sum())
study_days = max(1, df[df["最終学習日"] != "-"]["最終学習日"].str[:10].nunique())
total_difficulty = int(df[df["解答状況"] != "未着手"]["レベル"].sum())
memo_count = int(((df["つまずきメモ"].str.strip() != "") | (df["解法のコツ"].str.strip() != "")).sum())

cat_stage, cur_stage_name, next_stage_name, rem_pt, stage_pct = get_stage_info(total_earned)

# 上部ブランドバー
st.markdown("""
<div class="brand-bar">
    <div class="brand-logo">✍️ SIRUS</div>
    <div class="icon-pill"><span>↻</span><span>🔔</span><span>⚙️</span><span>⏱️</span></div>
</div>
""", unsafe_allow_html=True)

tab_home, tab_train, tab_review, tab_mypage = st.tabs([
    "🏠 ホーム",
    "🏋️ トレーニング",
    "📄 解き直し・メモ",
    "👤 マイページ"
])

# ==========================================
# タブ1：🏠 ホーム
# ==========================================
with tab_home:
    st.markdown(f'<div class="univ-header">🏛️ {st.session_state.target_univ}</div>', unsafe_allow_html=True)

    step_names = ["基礎固め", "得点力養成", "実戦力完成", "🔒 ???"]
    step_html = '<div class="stage-bar">'
    for s_name in step_names:
        is_act = "active" if (cat_stage == s_name or (cat_stage == "旧帝大突破" and s_name == "🔒 ???")) else ""
        label = "旧帝大突破" if (cat_stage == "旧帝大突破" and s_name == "🔒 ???") else s_name
        step_html += f'<div class="stage-step {is_act}">{label}</div>'
    step_html += "</div>"
    st.markdown(step_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:12px; color:#000000; font-weight:700; margin-bottom:2px;">到達ステージ（総合進度 {total_pct:.1f}%）</div>
    <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <div style="font-size:30px; font-weight:900; letter-spacing:1px;">{cur_stage_name}</div>
        <div style="font-size:28px; font-weight:900;">{total_earned:,} <span style="font-size:16px;">pt</span></div>
    </div>
    <div class="pill-progress-outer">
        <div class="pill-progress-inner" style="width: {max(6.0, stage_pct):.1f}%;"></div>
    </div>
    <div style="font-size:14px; font-weight:600; line-height:1.8; margin-bottom:14px;">
        次のステージ {next_stage_name} まで <span class="hl-yellow">あと {rem_pt:,} pt</span>
    </div>
    <hr style="border:none; border-top:1.5px solid #000000; margin:16px 0;">
    """, unsafe_allow_html=True)

    # 🎯 分野ごとの現在地
    chap_summary = df.groupby(["分野", "章"], sort=False).agg(
        問題数=("問題番号", "count"),
        完答=("解答状況", lambda s: (s == "解けた").sum()),
        獲得pt=("獲得pt", "sum"),
        目標pt=("満点pt", "sum")
    ).reset_index()
    chap_summary["進捗率(%)"] = (chap_summary["獲得pt"] / chap_summary["目標pt"] * 100).round(1)

    svg_chart = render_polar_chart_svg(chap_summary, target_ratio=0.72)
    st.markdown(f"""
    <div style="font-size:20px; font-weight:900; display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        🎯 分野ごとの現在地
    </div>
    <div style="display:flex; justify-content:center; gap:24px; font-size:12px; font-weight:700; color:#000000; margin-bottom:4px;">
        <span><span style="color:#C93B3B; font-weight:900;">- -</span> 今月の目標 (72%)</span>
        <span><span style="color:#000000; font-weight:900;">—</span> 合格ライン (100%: 10万pt)</span>
    </div>
    <div style="text-align:center; font-size:11px; color:#000000; margin-bottom:8px;">
        番号（1〜17）は『重要問題集』の第1章〜第16章・巻末補充問題に対応しています。
    </div>
    {svg_chart}
    """, unsafe_allow_html=True)

    selected_chap_idx = st.selectbox(
        "📌 扇形の番号（1〜17）から章の詳細を確認",
        options=list(range(1, 18)),
        format_func=lambda x: f"{x}. {chap_summary.iloc[x-1]['章']} （進捗: {chap_summary.iloc[x-1]['進捗率(%)']}%｜{chap_summary.iloc[x-1]['獲得pt']}/{chap_summary.iloc[x-1]['目標pt']}pt）"
    )

    review_df = df[df["解答状況"].isin(["一部解けた", "解けなかった"])]
    memo_df = df[(df["つまずきメモ"].str.strip() != "") | (df["解法のコツ"].str.strip() != "")]
    recent_date = pd.Timestamp.now().strftime("%m/%d")

    if not memo_df.empty:
        latest_memo_row = memo_df.iloc[-1]
        preview_txt = latest_memo_row["解法のコツ"] if latest_memo_row["解法のコツ"] else latest_memo_row["つまずきメモ"]
        preview_txt = (preview_txt[:28] + "…") if len(preview_txt) > 28 else preview_txt
        stack_preview = f"<strong>No.{int(latest_memo_row['問題番号'])} メモ:</strong><br>{preview_txt}"
    else:
        stack_preview = "「🏋️ トレーニング」で書いた解法のコツや弱点メモがここに表示されます。"

    st.markdown(f"""
    <div class="dotted-box">
        <div style="flex:1;">
            <div style="font-size:11px; font-weight:800; color:#000000;">総回答 / メモ数</div>
            <div style="font-size:24px; font-weight:900; margin-bottom:8px;">
                ✏️ <span class="hl-yellow">{answered_count}</span>
                <span style="font-size:13px; margin-left:4px;">(📝{memo_count})</span>
            </div>
            <div style="font-size:11px; font-weight:800; color:#000000;">累計日数</div>
            <div style="font-size:22px; font-weight:900; margin-bottom:8px;">📅 <span class="hl-yellow">{study_days}</span></div>
            <div style="font-size:11px; font-weight:800; color:#000000;">難易度合計</div>
            <div style="font-size:22px; font-weight:900;">📊 <span class="hl-yellow">{total_difficulty}</span></div>
        </div>
        <div class="stack-wrapper">
            <div class="stack-card" style="top:0px; left:24px; background:#F2F0E8;">
                <div class="stack-tab">見直し</div>
                <div class="date-badge" style="top:14px;">10/1</div>
            </div>
            <div class="stack-card" style="top:10px; left:12px; background:#F8F7F2;">
                <div class="stack-tab">見直し</div>
            </div>
            <div class="stack-card" style="top:20px; left:0px; background:#FFFFFF;">
                <div class="stack-tab">見直し</div>
                <div class="date-badge" style="top:36px;">{recent_date}</div>
                <div style="margin-top:8px; font-weight:800; color:#000000;">要復習: {len(review_df)}問</div>
                <div style="font-size:10px; color:#000000; margin-top:4px; line-height:1.35;">
                    {stack_preview}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# タブ2：🏋️ トレーニング
# ==========================================
with tab_train:
    st.subheader("🏋️ 『重要問題集』問題記録 ＆ 苦手克服メモ")

    q_number = st.number_input(
        "問題番号を選択 (No.1 〜 No.275)",
        min_value=1,
        max_value=275,
        value=int(st.session_state.selected_q),
        step=1
    )
    st.session_state.selected_q = q_number
    q_row = df[df["問題番号"] == q_number].iloc[0]

    st.markdown(f"""
    <div style="background:#FFFFFF; border:1.8px solid #000000; box-shadow:3px 4px 0px #000000; padding:14px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:12px; font-weight:800; color:#000000;">{q_row['分野']} ｜ {q_row['章']}</span>
            <span style="background:#7CE38B; border:1.5px solid #000000; border-radius:999px; padding:2px 10px; font-size:12px; font-weight:900; color:#000000;">
                満点 {q_row['満点pt']} pt
            </span>
        </div>
        <div style="font-size:28px; font-weight:900; margin:4px 0;">No.{q_number:02d} <span style="font-size:16px;">（{q_row['区分']}）</span></div>
        <div style="font-size:13px; font-weight:700;">
            現在の状態: <span class="hl-yellow">{q_row['解答状況']}</span>
            （獲得済: {q_row['獲得pt']} pt ／ 周回数: {q_row['周回数']}周 ／ 最終学習日: {q_row['最終学習日']}）
        </div>
    </div>
    """, unsafe_allow_html=True)

    status_choice = st.radio(
        "今回の解答状況を選択",
        ["解けた", "一部解けた", "解けなかった"],
        horizontal=True
    )
    has_shikou = st.checkbox("誌面に「思考」マークがある問題")

    st.markdown("#### 📝 解法のコツ ＆ つまずきメモ")
    existing_tags = [t for t in str(q_row["原因タグ"]).split(",") if t in CAUSE_TAGS]
    selected_tags = st.multiselect(
        "🏷️ つまずき原因・特徴タグ（複数選択可）",
        options=CAUSE_TAGS,
        default=existing_tags
    )

    trouble_memo = st.text_area(
        "❓ 分からなかった箇所・ミスした原因",
        value=str(q_row["つまずきメモ"]) if str(q_row["つまずきメモ"]) != "nan" else "",
        placeholder="例：(2)の圧平衡定数Kpの計算ミス。半透膜の浸透圧の公式を勘違いしていた。",
        height=80
    )

    tip_memo = st.text_area(
        "💡 解法のコツ・次回の着眼点",
        value=str(q_row["解法のコツ"]) if str(q_row["解法のコツ"]) != "nan" else "",
        placeholder="例：気体はまず物質量モルを確定する。芳香族の分離は塩基性・酸性の強弱順序に注意。",
        height=80
    )

    with st.expander("📸 問題画像のAI難易度判定（Lv.1〜10）を使う"):
        img_mode = st.radio("画像の入力方法", ["ファイル/写真選択", "スマホカメラで直接撮影"], horizontal=True)
        if img_mode == "スマホカメラで直接撮影":
            uploaded_img = st.camera_input("問題を撮影してください")
        else:
            uploaded_img = st.file_uploader("問題画像を選択", type=["png", "jpg", "jpeg"])

        current_lvl = int(q_row["レベル"])
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            if st.button("🔍 画像から難易度レベル(1〜10)を自動判定", use_container_width=True):
                current_lvl = evaluate_jumon_image(img, q_row["区分"], has_shikou)
                st.success(f"AI判定完了：No.{q_number} の難易度は **Lv.{current_lvl}** です！")

    final_lvl = st.slider("問題レベル (1〜10)", 1, 10, value=int(q_row["レベル"]))
    level_factor = 0.8 + (final_lvl / int(q_row["レベル"])) * 0.2
    new_earned_pt = int(round(q_row["満点pt"] * min(level_factor, 1.0) * STATUS_MULTIPLIER[status_choice]))

    if st.button(f"✅ No.{q_number} の学習結果とメモを保存（獲得: {new_earned_pt} pt）", type="primary", use_container_width=True):
        idx = df.index[df["問題番号"] == q_number][0]
        df.at[idx, "レベル"] = final_lvl
        df.at[idx, "解答状況"] = status_choice
        df.at[idx, "獲得pt"] = new_earned_pt
        df.at[idx, "周回数"] = int(q_row["周回数"]) + 1
        df.at[idx, "最終学習日"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        df.at[idx, "原因タグ"] = ",".join(selected_tags)
        df.at[idx, "つまずきメモ"] = trouble_memo.strip()
        df.at[idx, "解法のコツ"] = tip_memo.strip()
        save_data(df)
        st.toast(f"No.{q_number} を保存しました！")
        st.rerun()

# ==========================================
# タブ3：📄 解き直し・メモ
# ==========================================
with tab_review:
    st.subheader("🗂️ 苦手対策・解法コトノート")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_mode = st.selectbox(
            "表示フィルター",
            ["📝 メモがある問題のみ", "⚠️ 要復習（一部解けた・解けなかった）", "すべて表示"]
        )
    with col_f2:
        tag_filter = st.selectbox("原因タグで絞り込み", ["すべて"] + CAUSE_TAGS)

    keyword = st.text_input("🔍 キーワード検索")

    filtered_df = df.copy()
    if filter_mode == "📝 メモがある問題のみ":
        filtered_df = filtered_df[(filtered_df["つまずきメモ"].str.strip() != "") | (filtered_df["解法のコツ"].str.strip() != "") | (filtered_df["原因タグ"].str.strip() != "")]
    elif filter_mode == "⚠️ 要復習（一部解けた・解けなかった）":
        filtered_df = filtered_df[filtered_df["解答状況"].isin(["一部解けた", "解けなかった"])]

    if tag_filter != "すべて":
        filtered_df = filtered_df[filtered_df["原因タグ"].str.contains(tag_filter, na=False)]

    if keyword.strip():
        kw = keyword.strip()
        filtered_df = filtered_df[
            filtered_df["つまずきメモ"].str.contains(kw, case=False, na=False)
            | filtered_df["解法のコツ"].str.contains(kw, case=False, na=False)
            | filtered_df["章"].str.contains(kw, case=False, na=False)
        ]

    if filtered_df.empty:
        st.info("条件に一致するメモ・復習問題はまだありません。")
    else:
        st.caption(f"該当件数: **{len(filtered_df)} 問**")
        for _, m_row in filtered_df.iterrows():
            q_no = int(m_row["問題番号"])
            tags_list = [t for t in str(m_row["原因タグ"]).split(",") if t]
            tags_html = "".join([f'<span class="memo-tag-chip">🏷️ {t}</span>' for t in tags_list])

            trouble_html = (
                f'<div class="memo-box-trouble"><strong>❓ つまずき原因:</strong><br>{m_row["つまずきメモ"]}</div>'
                if str(m_row["つまずきメモ"]).strip() else ""
            )
            tip_html = (
                f'<div class="memo-box-tip"><strong>💡 解法のコツ:</strong><br>{m_row["解法のコツ"]}</div>'
                if str(m_row["解法のコツ"]).strip() else ""
            )

            st.markdown(f"""
            <div class="memo-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <span style="font-size:11px; font-weight:800; color:#000000;">{m_row['分野']} ｜ {m_row['章']}</span>
                    <span style="font-size:11px; font-weight:800; background:#EEECE2; border:1px solid #000000; padding:1px 8px; border-radius:999px; color:#000000;">
                        {m_row['解答状況']}（{m_row['周回数']}周）
                    </span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <div style="font-size:22px; font-weight:900;">No.{q_no:02d} <span style="font-size:13px;">（{m_row['区分']} / Lv.{m_row['レベル']}）</span></div>
                    <div style="font-size:11px; color:#000000;">最終学習: {m_row['最終学習日']}</div>
                </div>
                <div style="margin-top:4px;">{tags_html}</div>
                {trouble_html}
                {tip_html}
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"✏️ No.{q_no:02d} をトレーニングで開く", key=f"edit_memo_{q_no}"):
                st.session_state.selected_q = q_no
                st.toast(f"No.{q_no} をセットしました！")

# ==========================================
# タブ4：👤 マイページ
# ==========================================
with tab_mypage:
    st.subheader("👤 志望校設定 ＆ バックアップ管理")
    new_univ = st.text_input("目標大学・学部名", value=st.session_state.target_univ)
    if new_univ != st.session_state.target_univ:
        st.session_state.target_univ = new_univ
        st.rerun()

    st.markdown("#### 📊 分野別 進捗率 (%)")
    field_grp = df.groupby("分野", sort=False).agg({"獲得pt": "sum", "満点pt": "sum"}).reset_index()
    field_grp["進捗率(%)"] = (field_grp["獲得pt"] / field_grp["満点pt"] * 100).round(2)
    st.bar_chart(field_grp.set_index("分野")["進捗率(%)"])

    st.divider()
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 進捗・メモデータのバックアップ (CSV)",
        data=csv_bytes,
        file_name="jumon_progress_backup.csv",
        mime="text/csv",
        use_container_width=True
    )
    restore_file = st.file_uploader("📤 バックアップから復元", type=["csv"])
    if restore_file is not None:
        if st.button("データを復元する", use_container_width=True):
            df_restored = pd.read_csv(restore_file)
            save_data(df_restored)
            st.success("復元しました！")
            st.rerun()