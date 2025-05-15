# streamlit_ttest_plot.py (CSV 기반 버전, OCR 제거)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

st.set_page_config(layout="wide")

# 📝 사용자 안내 텍스트 ---------------------------------------------
st.title("Paired t-test Boxplot Visualizer from CSV File")

st.markdown("""
## ✅ 사용법 안내 (Windows 기준)

1. **엑셀에서 원하는 시트**를 선택하세요.
2. 상단 메뉴에서 **파일 > 다른 이름으로 저장** 또는 **F12** 키를 누르세요.
3. **파일 형식**을 `CSV UTF-8 (쉼표로 분리) (*.csv)` 로 설정하세요.
4. 저장한 `.csv` 파일을 이 앱에 업로드하면 자동으로 시각화됩니다.

**예시 열 구성:**
- `Timepoint`, `Mean`, `SD`, `N`, `p_value`
""")

# 📁 CSV 업로드 ------------------------------------------------------
uploaded_file = st.file_uploader("📂 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Step 1: 업로드된 데이터 미리보기")
    st.dataframe(df)

    # 시뮬레이션된 raw data 생성 --------------------------------------
    st.subheader("Step 2: Boxplot을 위한 가상 Raw 데이터 생성")
    raw_data = pd.DataFrame([
        (row["Timepoint"], np.random.normal(row["Mean"], row["SD"], int(row["N"])))
        for _, row in df.iterrows()
    ], columns=["Timepoint", "Value"]).explode("Value")
    raw_data["Value"] = raw_data["Value"].astype(float)
    st.write(raw_data.head())

    # 📊 박스플롯 시각화 --------------------------------------------
    st.subheader("Step 3: Boxplot 시각화")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.set_title("Boxplot per Timepoint with Statistical Significance", fontsize=14, pad=20)

    timepoints = df["Timepoint"].tolist()
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, rotation=45)

    sns.boxplot(data=raw_data, x="Timepoint", y="Value", hue="Timepoint", palette="pastel", ax=ax, legend=False, boxprops=dict(edgecolor='none'))
    sns.stripplot(data=raw_data, x="Timepoint", y="Value", color='gray', alpha=0.3, jitter=True, ax=ax)

    for i, row in df.iterrows():
        if not pd.isna(row["p_value"]):
            label = "Significant (p<.05)" if row["p_value"] < 0.05 else "Not significant"
            y_pos = raw_data[raw_data["Timepoint"] == row["Timepoint"]]["Value"].max() + 1.5
            ax.text(i, y_pos, label, ha='center', fontsize=10, color="black")

    ax.set_ylabel("Measurement Value")
    ax.set_xlabel("Timepoint")
    fig.tight_layout(pad=3.0)

    st.pyplot(fig)

    # 📥 다운로드 버튼 ------------------------------------------------
    st.subheader("Step 4: 시각화 결과 다운로드")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Download Plot as PNG",
        data=buf.getvalue(),
        file_name="paired_ttest_boxplot.png",
        mime="image/png"
    )

    # 🎉 마무리 메시지 ----------------------------------------------
    st.success("시각화가 성공적으로 완료되었습니다!")
    st.balloons()
    st.markdown("### 🎉 허유란 선생님 화이팅!")

else:
    st.warning("먼저 .csv 파일을 업로드해주세요. (엑셀에서 시트를 .csv로 저장하는 방법은 위 안내 참고)")
