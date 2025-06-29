import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import warnings
import math

warnings.filterwarnings("ignore")

# ─── (1) 페이지 설정 ────────────────────────────────────────
st.set_page_config(
    page_title="실시간 전기요금 예측 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── (2) 깔끔한 화이트 테마 CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Noto Sans KR', sans-serif;
    }
    
    .stApp {
        background-color: #fafbfc;
    }
    
    /* 메인 헤더 */
    .main-header {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8eaed;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #1a73e8;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header .subtitle {
        color: #5f6368;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* KPI 카드 */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .kpi-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        transform: translateY(-1px);
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #5f6368;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        color: #202124;
        font-weight: 700;
        line-height: 1;
    }
    
    .kpi-unit {
        font-size: 0.75rem;
        color: #80868b;
        margin-top: 0.25rem;
    }
    
    /* 차트 컨테이너 */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        margin-bottom: 1rem;
    }
    
    .chart-title {
        font-size: 1.1rem;
        color: #202124;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f3f4;
    }
    
    /* 테이블 스타일 */
    .table-container {
        background: white;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        overflow: hidden;
    }
    
    .table-header {
        background: #f8f9fa;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e8eaed;
    }
    
    .table-title {
        font-size: 1.1rem;
        color: #202124;
        font-weight: 600;
        margin: 0;
    }
    
    /* 네비게이션 버튼 */
    .nav-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.5rem;
        background: #f8f9fa;
        border-bottom: 1px solid #e8eaed;
    }
    
    .nav-info {
        font-size: 0.9rem;
        color: #5f6368;
        font-weight: 500;
    }
    
    /* 설명 카드 */
    .info-card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1a73e8;
        margin: 0.5rem 0;
    }
    
    .info-card .info-title {
        font-size: 0.9rem;
        color: #1a73e8;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-card .info-content {
        font-size: 0.85rem;
        color: #5f6368;
        line-height: 1.4;
    }
    
    /* 상태 표시 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-running {
        background: #e8f5e8;
        color: #137333;
    }
    
    .status-stopped {
        background: #fce8e6;
        color: #d93025;
    }
    
    /* 사이드바 스타일 */
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border: 1px solid #e8eaed;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        color: #202124;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f3f4;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: #1a73e8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    
    .stButton > button:hover {
        background: #1557b0;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
    }
    
    /* Plotly 차트 스타일링 */
    .js-plotly-plot {
        border-radius: 6px;
    }
    
    /* 데이터프레임 스타일링 */
    .stDataFrame {
        border: none !important;
    }
    
    .stDataFrame > div {
        border: none !important;
        box-shadow: none !important;
    }
    
    /* 숨기고 싶은 요소 */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* header {
        visibility: hidden;
    }
    */
</style>
""", unsafe_allow_html=True)

# ─── SHAP 관련 헬퍼 함수 ─────────────────────────────────────
def generate_dummy_shap_values():
    feature_names = st.session_state.feat_data.columns.drop(
        ["id", "측정일시", "target"]
    ).tolist()
    np.random.seed(len(st.session_state.shap_history))
    vals = np.random.randn(len(feature_names))
    return dict(zip(feature_names, vals))

def create_shap_chart():
    if not st.session_state.shap_history:
        return None

    selected_features = [
        "전력사용량(kWh)",
        "지상무효전력량(kVarh)",
        "진상무효전력량(kVarh)",
        "탄소배출량(tCO2)",
        "진상역률_이진",
        "지상역률_이진",
    ]

    mean_abs = {
        f: np.mean([abs(h[f]) for h in st.session_state.shap_history])
        for f in selected_features
        if f in st.session_state.shap_history[0]
    }

    feats_sorted = sorted(mean_abs.items(), key=lambda x: x[1], reverse=True)
    top_feats = [k for k, _ in feats_sorted]
    top_vals = [v for _, v in feats_sorted]

    fig = go.Figure(
        go.Bar(
            x=top_vals[::-1],
            y=top_feats[::-1],
            orientation="h",
            marker_color="#1a73e8",
            marker_line=dict(width=0),
            hovertemplate="<b>%{y}</b><br>평균 |SHAP|: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="",
        xaxis_title="평균 |SHAP|",
        yaxis_title="",
        height=350,
        template="plotly_white",
        margin=dict(l=120, r=20, t=20, b=40),
        font=dict(family="Noto Sans KR", size=11),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return fig

# ─── 세션 상태 초기화 ────────────────────────────────────────
def init_state():
    if "data" not in st.session_state:
        st.session_state.data = pd.read_csv(
            "streamlit_/models/final_lstm_target.csv", parse_dates=["측정일시"]
        )
    if "feat_data" not in st.session_state:
        st.session_state.feat_data = pd.read_csv(
            "streamlit_/models/target_pred_feature_lstm.csv", parse_dates=["측정일시"]
        )
    if "start_idx" not in st.session_state:
        df = st.session_state.data
        matches = df.index[df["id"] == 32111].tolist()
        st.session_state.start_idx = matches[0] if matches else 0
        st.session_state.idx = st.session_state.start_idx
    for key in ["time_list", "cost_list", "shap_history"]:
        st.session_state.setdefault(key, [])
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("page", 0)

init_state()

# ─── 사이드바 제어판 ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    /* 모든 버튼 기본 회색 테두리 스타일 */
    div[data-testid="stButton"] > button {
        background-color: rgba(95, 99, 104, 0.05) !important;
        border: 2px solid #5f6368 !important;
        color: #5f6368 !important;
        font-weight: 600;
        border-radius: 6px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }

    /* hover 또는 클릭 시 강조 효과 */
    div[data-testid="stButton"] > button:hover {
        background-color: rgba(95, 99, 104, 0.15) !important;
        box-shadow: 0 0 0 3px rgba(95, 99, 104, 0.2);
    }

    /* 눌렀을 때(active)는 hover와 동일하게 보이도록 처리 */
    div[data-testid="stButton"] > button:active {
        background-color: rgba(95, 99, 104, 0.25) !important;
        box-shadow: 0 0 0 3px rgba(95, 99, 104, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ 시스템 제어")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
        if st.button("시작", key="start_btn"):
            st.session_state.running = True
    with btn_col2:
        if st.button("정지", key="stop_btn"):
            st.session_state.running = False
    with btn_col3:
        if st.button("리셋", key="reset_btn"):
            st.session_state.idx = st.session_state.start_idx
            st.session_state.time_list.clear()
            st.session_state.cost_list.clear()
            st.session_state.shap_history.clear()
            st.session_state.running = False
            st.session_state.page = 0

    if st.session_state.running:
        status_class = "status-running"
        status_text = "🟢 실행 중"
    else:
        status_class = "status-stopped"
        status_text = "🔴 정지됨"

    st.markdown(f"""
    <div style="margin-top: 1rem;">
        <div class="status-indicator {status_class}" style="width: 100%; text-align: center;">
            {status_text}
        </div>
    </div>
    """, unsafe_allow_html=True)




# ─── 메인 화면 함수 ─────────────────────────────────────────
def show_main():
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>실시간 전기요금 모니터링</h1>
        <div class="subtitle">전력 사용량 분석 및 요금 예측 대시보드</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI 카드들
    start, idx = st.session_state.start_idx, st.session_state.idx
    df_slice = st.session_state.feat_data.iloc[start:idx]
    
    total_cost = sum(st.session_state.cost_list)
    total_kwh = df_slice["전력사용량(kWh)"].sum()
    total_kvarh_jisang = df_slice["지상무효전력량(kVarh)"].sum()
    total_kvarh_jinsang = df_slice["진상무효전력량(kVarh)"].sum()
    total_co2 = df_slice["탄소배출량(tCO2)"].sum()

    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">누적 전기요금</div>
            <div class="kpi-value">₩{total_cost:,.0f}</div>
            <div class="kpi-unit">원</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">누적 전력사용량</div>
            <div class="kpi-value">{total_kwh:.1f}</div>
            <div class="kpi-unit">kWh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">지상무효전력량</div>
            <div class="kpi-value">{total_kvarh_jisang:.1f}</div>
            <div class="kpi-unit">kVarh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">진상무효전력량</div>
            <div class="kpi-value">{total_kvarh_jinsang:.1f}</div>
            <div class="kpi-unit">kVarh</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">탄소배출량</div>
            <div class="kpi-value">{total_co2:.2f}</div>
            <div class="kpi-unit">tCO₂</div>
        </div>
        """, unsafe_allow_html=True)

    # 차트 섹션
    chart_col1, chart_col2 = st.columns([3, 2])
    
    with chart_col1:
        
        df_plot = pd.DataFrame({
            "측정일시": st.session_state.time_list,
            "전기요금(원)": st.session_state.cost_list,
        })
        
        if not df_plot.empty:
            fig = px.line(
                df_plot,
                x="측정일시",
                y="전기요금(원)",
                markers=True,
                line_shape="linear"
            )
            fig.update_traces(
                line_color="#1a73e8",
                marker_color="#1a73e8",
                marker_size=6,
                line_width=3
            )
            fig.update_layout(
                 title=dict(
                text="실시간 전기요금 추이",
                x=0.5,
                y=0.9,            # 기본 1.0보다 작게 주면 아래로 내려옵니다
                yanchor="top",          # 가운데 정렬
                xanchor="center",
                font=dict(size=20)   # 정렬 기준
                ),
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=40, b=40),
                font=dict(family="Noto Sans KR", size=11),
                xaxis_title="측정일시",
                yaxis_title="전기요금 (원)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(gridcolor="#f1f3f4"),
                yaxis=dict(gridcolor="#f1f3f4")
            )
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
        else:
            st.info("데이터가 수집되는 중입니다...")

    with chart_col2:
        shap_fig = create_shap_chart()
        if shap_fig:
            shap_fig.update_layout(
                title=dict(
                    text=" 누적 SHAP 중요도",
                    x=0.5,              # paper 기준 50%
                    xanchor="center",   # 중앙 정렬 기준점을 가운데로
                    y=0.95,             # (선택) 위쪽으로 살짝 올리고 싶으면 1.0에 가깝게
                    yanchor="top"
                ),
                title_font_size=18,
                template="plotly_white",
                margin=dict(l=40, r=40, t=60, b=40),  # 좌우 마진을 동일하게
            )
            st.plotly_chart(shap_fig, use_container_width=True, key="shap_chart")
            # 누적 SHAP 설명 토글 추가
            with st.expander("누적 SHAP 중요도 설명"):
                st.write("실시간으로 수집되는 전기요금 데이터에 대해, 각 변수의 SHAP 값 절대값을 평균내어 해당 변수가 전기요금에 미치는 전체적인 영향력을 확인할 수 있습니다.\n"
                        "값이 클수록 전기요금에 더 큰 영향을 주는 변수임을 의미합니다.")

        else:
            st.info("SHAP 분석 준비 중...")
    # 하단 섹션
    bottom_col1, bottom_col2 = st.columns([3, 2])

    with bottom_col1:
    # 테이블 헤더 (HTML)
        st.markdown("""
            <div class="table-container">
                <div class="table-header" style="text-align: center; width: 100%;">
                    <div class="table-title">실시간 데이터</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 원본 데이터 슬라이스
        df_slice = st.session_state.feat_data.iloc[
            st.session_state.start_idx : st.session_state.idx
        ].reset_index(drop=True)

        if not df_slice.empty:
            # 페이징 파라미터
            total_rows = len(df_slice)
            page_size = 8
            total_pages = max(math.ceil(total_rows / page_size), 1)

            # 페이지 이동 함수
            def go_prev():
                st.session_state.page = max(0, st.session_state.page - 1)
            def go_next():
                st.session_state.page = min(total_pages - 1, st.session_state.page + 1)

            # 현재 페이지 데이터
            start_idx = st.session_state.page * page_size
            end_idx = start_idx + page_size
            show_cols = [
                "측정일시", "전력사용량(kWh)", "지상무효전력량(kVarh)",
                "진상무효전력량(kVarh)", "탄소배출량(tCO2)", "진상역률_이진", "지상역률_이진"
            ]
            display_df = df_slice[show_cols].iloc[start_idx:end_idx]
            display_df["측정일시"] = display_df["측정일시"].dt.strftime("%Y-%m-%d %H:%M")
            

            # 표 출력
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ─── 표 아래에 네비게이션 ─────────────────────────────────
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            with nav_col1:
                st.button("◀ 이전",
                          disabled=(st.session_state.page <= 0),
                          on_click=go_prev,
                          key="prev_btn")
            with nav_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; color: #5f6368; font-size: 0.9rem;">
                    페이지 {st.session_state.page + 1} / {total_pages}
                </div>
                """, unsafe_allow_html=True)
            with nav_col3:
                st.button("다음 ▶",
                          disabled=(st.session_state.page >= total_pages - 1),
                          on_click=go_next,
                          key="next_btn")
        else:
            st.info("데이터가 로드되는 중입니다...")


        # 설명 카드들
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("""
            <div class="info-card">
                <div class="info-title">진상역률_이진</div>
                <div class="info-content">
                    1 = 역률 기준(95%) 이상<br>
                    0 = 기준 미만
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with exp_col2:
            st.markdown("""
            <div class="info-card">
                <div class="info-title">지상역률_이진</div>
                <div class="info-content">
                    1 = 역률 기준(65%) 이상<br>
                    0 = 기준 미만
                </div>
            </div>
            """, unsafe_allow_html=True)

    with bottom_col2:
    
        if st.session_state.shap_history:
            last_shap = st.session_state.shap_history[-1]
            show_feats = [
                "전력사용량(kWh)", "지상무효전력량(kVarh)", "진상무효전력량(kVarh)",
                "탄소배출량(tCO2)", "진상역률_이진", "지상역률_이진"
            ]
            feats = [f for f in show_feats if f in last_shap]
            vals = [last_shap[f] for f in feats]
            colors = ["#ea4335" if v > 0 else "#1a73e8" for v in vals]

            fig = go.Figure(
                go.Bar(
                    x=vals[::-1],
                    y=feats[::-1],
                    orientation="h",
                    marker_color=colors[::-1],
                    marker_line=dict(width=0),
                    hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>",
                )
            )
            fig.update_layout(
                title=dict(
                text="최근 SHAP 기여도",
                x=0.5,            # paper 좌표계 중앙
                xanchor="center", # 중앙 anchor
                font=dict(size=18)
                ),
                xaxis_title="SHAP 값",
                yaxis_title="",
                height=350,
                template="plotly_white",
                margin=dict(l=120, r=20, t=60, b=40),
                font=dict(family="Noto Sans KR", size=11),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
            st.plotly_chart(fig, use_container_width=True, key="recent_shap")
            # 누적 SHAP 설명 토글 추가
            with st.expander("실시간 SHAP 기여도 설명"):
                st.write("실시간으로 유입되는 각 변수들이 전기요금에 긍정적(상승) 또는 부정적(하락)으로 얼마나 기여했는지 그 영향을 확인할 수 있습니다."
                         "변수별로 전기요금 예측값을 높였는지, 낮췄는지 직관적으로 파악할 수 있습니다.")
        else:
            st.info("SHAP 분석 데이터가 없습니다.")

# ─── 메인 실행 루프 ─────────────────────────────────────────
if st.session_state.running:
    df = st.session_state.data
    if st.session_state.idx < len(df):
        row = df.iloc[st.session_state.idx]
        st.session_state.time_list.append(row["측정일시"])
        st.session_state.cost_list.append(row["target"])
        st.session_state.idx += 1

        new_shap = generate_dummy_shap_values()
        st.session_state.shap_history.append(new_shap)

        show_main()
        time.sleep(10)
        st.rerun()
    else:
        st.warning("⚠️ 더 이상 불러올 데이터가 없습니다.")
        st.session_state.running = False
        show_main()
else:
    if st.session_state.time_list:
        show_main()
    else:
        st.markdown("""
        <div class="main-header">
            <h1>실시간 전기요금 예측 시스템</h1>
            <div class="subtitle">AI 기반 전력 사용량 분석 및 요금 예측 대시보드</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: white; border-radius: 12px; 
             box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e8eaed;">
            <h3 style="color: #5f6368; margin-bottom: 1rem;">시스템 대기 중</h3>
            <p style="color: #80868b; margin-bottom: 2rem;">
                좌측 사이드바에서 <strong>▶️ 시작</strong> 버튼을 클릭하여<br>
                실시간 전기요금 모니터링을 시작하세요.
            </p>
        </div>
        """, unsafe_allow_html=True)