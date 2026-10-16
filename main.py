import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 끝
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    # 날짜 문자열 앞뒤 공백/탭 제거 후 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    # 2025년까지만 사용
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일 수(날짜 기준)와 평균기온
    yearly = (
        df.groupby("연도")
        .agg(관측일=("날짜", "nunique"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 만든 회귀 직선을 이용해, 고른 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

# 회귀 직선과 상관계수
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 연도 선택 슬라이더 ----
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
      <div style="font-size:1.2rem; color:gray;">{year}년 서울의 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:700;">{pred:.1f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if year < start_year or year > end_year:
    st.caption(
        f"※ {year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이라, "
        "직선을 연장해서 계산한 값입니다."
    )

# ---- 데이터 요약 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

st.write(
    f"회귀 직선: **기온 = {slope:.4f} × 연도 + {intercept:.2f}**  "
    f"(10년마다 약 {slope * 10:.2f}°C 변화)"
)
st.caption(
    f"{LAST_YEAR}년까지의 자료만 사용했고, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# ---- plotly 그래프: 산점도 + 회귀 직선 + 선택한 연도 ----
line_x0 = min(start_year, year)
line_x1 = max(end_year, year)
line_x = np.array([line_x0, line_x1], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    height=520,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        yearly.rename(columns={"관측일": "관측일 수"}).round({"연평균기온": 2}),
        use_container_width=True,
        hide_index=True,
    )
