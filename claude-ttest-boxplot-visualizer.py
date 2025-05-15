import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import re

st.set_page_config(layout="wide")
st.title("스크린샷에서 대응표본 t-test 박스플롯 시각화")

st.markdown("""
이 앱을 사용하여 다음을 수행할 수 있습니다:
1. 스크린샷 이미지 업로드 (예: SPSS 출력 또는 엑셀 스크린샷).
2. OCR을 사용하여 숫자 값 추출.
3. 평균, 표준편차, N을 사용하여 데이터 시뮬레이션.
4. 타임포인트별 박스플롯 및 유의성 표시.
""")

# 사이드바에 OCR 언어 선택 옵션 추가
ocr_lang = st.sidebar.selectbox(
    "OCR 언어 선택",
    options=["eng", "eng+kor", "kor+eng"],
    index=1
)

# 사이드바에 추가 설정
st.sidebar.subheader("추가 설정")
show_raw_text = st.sidebar.checkbox("원본 OCR 텍스트 표시", value=True)
manual_override = st.sidebar.checkbox("수동으로 데이터 입력", value=False)
show_strip_plot = st.sidebar.checkbox("개별 데이터 포인트 표시", value=True)
significance_threshold = st.sidebar.slider("유의성 임계값 (p-value)", 
                                          min_value=0.001, max_value=0.1, value=0.05, step=0.001, 
                                          format="%.3f")

uploaded_file = st.file_uploader("t-test 결과 스크린샷 업로드 (PNG/JPG)", type=["png", "jpg", "jpeg"])

def extract_numeric_values(text):
    """OCR 텍스트에서 정규식을 사용하여 숫자 값 추출"""
    # 숫자 추출 패턴 (소수점 포함)
    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    return [float(num) for num in numbers]

def parse_table_from_text(text):
    """OCR 텍스트에서 테이블 구조 파싱 시도"""
    lines = text.strip().split('\n')
    
    # 데이터를 저장할 리스트
    timepoints = []
    means = []
    sds = []
    ns = []
    p_values = []
    
    # 텍스트에서 "Mean", "SD", "N", "p" 등의 키워드가 있는 라인 찾기
    header_line = None
    for i, line in enumerate(lines):
        if re.search(r'Mean|SD|N|p.?value', line, re.IGNORECASE):
            header_line = i
            break
    
    if header_line is not None:
        # 헤더 이후의 라인을 처리
        for line in lines[header_line+1:]:
            # 숫자만 추출
            nums = extract_numeric_values(line)
            if len(nums) >= 3:  # 최소한 Mean, SD, N이 있어야 함
                # 첫 번째 숫자가 타임포인트 번호일 수 있음
                if len(timepoints) < 7:  # 최대 7개의 타임포인트만 처리
                    if len(nums) >= 4:  # p-value가 있는 경우
                        timepoints.append(f"T{len(timepoints)+1}")
                        means.append(nums[0])
                        sds.append(nums[1])
                        ns.append(int(nums[2]))
                        p_values.append(nums[3] if nums[3] <= 1 else None)  # p-value는 0~1 사이여야 함
                    else:
                        timepoints.append(f"T{len(timepoints)+1}")
                        means.append(nums[0])
                        sds.append(nums[1])
                        ns.append(int(nums[2]))
                        p_values.append(None)
    
    # 타임포인트 이름이 별도로 있는지 확인
    timepoint_names = []
    for line in lines:
        # 타임포인트 이름으로 보이는 패턴 (NB, NA1 등)
        match = re.findall(r'\b([A-Z]{1,2}\d*|[A-Z]{1,2}[_-]?\d*)\b', line)
        if match and len(match) >= 2:
            timepoint_names = match
            break
    
    # 타임포인트 이름이 추출되었고, 개수가 맞으면 사용
    if timepoint_names and len(timepoint_names) >= len(timepoints):
        timepoints = timepoint_names[:len(timepoints)]
    
    # 데이터가 충분히 추출되었는지 확인
    if len(timepoints) >= 2 and len(means) == len(timepoints):
        data = {
            "Timepoint": timepoints,
            "Mean": means,
            "SD": sds,
            "N": ns,
            "p_value": p_values
        }
        return pd.DataFrame(data)
    
    # 데이터 추출이 실패한 경우 샘플 데이터 반환
    return None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # OCR을 사용하여 텍스트 추출
    with st.spinner('이미지에서 텍스트를 추출 중입니다...'):
        text = pytesseract.image_to_string(image, lang=ocr_lang)
    
    if show_raw_text:
        st.subheader("1단계: OCR 추출 텍스트")
        st.text(text)
    
    # OCR 텍스트에서 데이터 추출 시도
    extracted_df = parse_table_from_text(text)
    
    # 수동 데이터 입력 또는 샘플 데이터 사용
    if manual_override or extracted_df is None:
        st.subheader("2단계: 수동으로 데이터 입력")
        
        # 샘플 데이터 설정
        sample_data = {
            "Timepoint": ["NB", "NA1", "NA2", "NA3", "NA4", "NA5", "NFU"],
            "Mean": [104.6667, 105.2286, 105.1810, 104.6667, 102.8571, 102.9810, 106.1692],
            "SD": [6.7413, 6.4390, 6.1071, 6.7413, 6.9111, 6.9561, 6.2758],
            "N": [42, 42, 42, 42, 42, 42, 26],
            "p_value": [None, 0.024, 0.106, 0.008, 0.001, 0.000, 0.004]
        }
        
        # 수동 입력을 위한 인터페이스
        col1, col2 = st.columns(2)
        
        with col1:
            num_timepoints = st.number_input("타임포인트 개수", min_value=2, max_value=10, value=7)
        
        # 데이터프레임 편집 가능한 형태로 표시
        edited_df = pd.DataFrame({
            "Timepoint": sample_data["Timepoint"][:num_timepoints],
            "Mean": sample_data["Mean"][:num_timepoints],
            "SD": sample_data["SD"][:num_timepoints],
            "N": sample_data["N"][:num_timepoints],
            "p_value": sample_data["p_value"][:num_timepoints]
        })
        
        edited_df = st.data_editor(edited_df, use_container_width=True, 
                                  column_config={
                                      "Timepoint": st.column_config.TextColumn("Timepoint", help="타임포인트 이름"),
                                      "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                                      "SD": st.column_config.NumberColumn("SD", format="%.4f"),
                                      "N": st.column_config.NumberColumn("N", format="%d"),
                                      "p_value": st.column_config.NumberColumn("p-value", format="%.4f")
                                  })
        df = edited_df
    else:
        st.subheader("2단계: 추출된 데이터")
        st.dataframe(extracted_df)
        df = extracted_df
    
    st.subheader("3단계: 박스플롯을 위한 시뮬레이션 원시 데이터")
    
    # 난수 생성을 위한 시드 설정 (일관된 결과 위해)
    seed = st.sidebar.number_input("난수 시드", min_value=0, max_value=9999, value=42)
    np.random.seed(seed)
    
    # 각 타임포인트별 데이터 생성
    raw_data_list = []
    for _, row in df.iterrows():
        # N개의 샘플 생성
        samples = np.random.normal(row["Mean"], row["SD"], int(row["N"]))
        # 데이터프레임 형태로 저장
        for sample in samples:
            raw_data_list.append({"Timepoint": row["Timepoint"], "Value": sample})
    
    raw_data = pd.DataFrame(raw_data_list)
    st.write(raw_data.head())
    
    # Matplotlib 한글 폰트 설정 (맥에서는 AppleGothic 사용)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    st.subheader("4단계: 박스플롯 시각화")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 박스플롯 그리기
    sns.boxplot(data=raw_data, x="Timepoint", y="Value", palette="pastel", ax=ax)
    
    # 개별 데이터 포인트 표시 (선택적)
    if show_strip_plot:
        sns.stripplot(data=raw_data, x="Timepoint", y="Value", color='gray', alpha=0.3, jitter=True, ax=ax)
    
    # y축 범위 결정
    y_values = raw_data["Value"].dropna()
    if not y_values.empty:
        y_min = y_values.min()
        y_max = y_values.max()
        # 텍스트 표시를 위한 여백
        text_offset = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
    else:
        y_min, y_max = 0, 10
        text_offset = 0.5
    
    # 각 타임포인트별 통계적 유의성 표시
    for i, row in df.iterrows():
        if row["p_value"] is not None:
            # p-value에 따라 라벨 설정
            if row["p_value"] <= significance_threshold:
                label = f"유의함 (p={row['p_value']:.3f})"
                color = "red"
            else:
                label = f"유의하지 않음 (p={row['p_value']:.3f})"
                color = "black"
            
            # 현재 타임포인트의 최대값 찾기
            current_values = raw_data[raw_data["Timepoint"] == row["Timepoint"]]["Value"].dropna()
            y_pos = y_max + text_offset  # 기본값
            
            if not current_values.empty:
                current_max = current_values.max()
                if not pd.isna(current_max):
                    y_pos = current_max + text_offset
            
            # 해당 위치에 텍스트 추가
            ax.text(i, y_pos, label, ha='center', fontsize=10, color=color, rotation=45)
    
    # 그래프 제목 및 축 레이블 설정
    title_obj = ax.set_title("타임포인트별 박스플롯 및 통계적 유의성", fontsize=14)
    title_obj.set_y(1.05)  # 제목 아래 여백 추가
    
    ax.set_ylabel("측정값")
    ax.set_xlabel("타임포인트")
    
    # x축 눈금 설정
    ax.set_xticks(range(len(df["Timepoint"])))
    ax.set_xticklabels(df["Timepoint"], rotation=45, ha="right")
    
    # 모든 텍스트가 보이도록 y축 범위 조정
    ax.autoscale(enable=True, axis='y', tight=False)
    current_ylim_min, current_ylim_max = ax.get_ylim()
    
    # 텍스트 위치를 기반으로 y축 최대값 조정
    max_text_y_coord = current_ylim_max
    if ax.texts:
        text_y_positions = [t.get_position()[1] for t in ax.texts 
                          if t.get_position()[1] is not None]
        if text_y_positions:
            max_text_y_coord = max(text_y_positions)
    
    # 최종 y축 범위 설정
    ax.set_ylim(current_ylim_min, max(current_ylim_max, max_text_y_coord + text_offset * 2.0))
    
    # 레이아웃 조정
    fig.tight_layout(pad=1.1)
    st.pyplot(fig)
    
    # 데이터 다운로드 옵션 제공
    col1, col2 = st.columns(2)
    
    with col1:
        csv_stats = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="통계 데이터 CSV 다운로드",
            data=csv_stats,
            file_name='t_test_stats.csv',
            mime='text/csv',
        )
    
    with col2:
        csv_raw = raw_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="원시 데이터 CSV 다운로드",
            data=csv_raw,
            file_name='simulated_raw_data.csv',
            mime='text/csv',
        )
    
    st.success("시각화가 완료되었습니다!")
else:
    st.info("시작하려면 스크린샷을 업로드하세요.")
    
    # 샘플 이미지 표시
    st.subheader("예시 출력")
    
    # 샘플 데이터로 데모 차트 표시
    sample_data = {
        "Timepoint": ["NB", "NA1", "NA2", "NA3", "NA4", "NA5", "NFU"],
        "Mean": [104.6667, 105.2286, 105.1810, 104.6667, 102.8571, 102.9810, 106.1692],
        "SD": [6.7413, 6.4390, 6.1071, 6.7413, 6.9111, 6.9561, 6.2758],
        "N": [42, 42, 42, 42, 42, 42, 26],
        "p_value": [None, 0.024, 0.106, 0.008, 0.001, 0.000, 0.004]
    }
    df = pd.DataFrame(sample_data)
    
    # 난수 생성기 시드 설정
    np.random.seed(42)
    
    # 가상의 원시 데이터 생성
    raw_data_list = []
    for _, row in df.iterrows():
        samples = np.random.normal(row["Mean"], row["SD"], int(row["N"]))
        for sample in samples:
            raw_data_list.append({"Timepoint": row["Timepoint"], "Value": sample})
    
    raw_data = pd.DataFrame(raw_data_list)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=raw_data, x="Timepoint", y="Value", palette="pastel", ax=ax)
    sns.stripplot(data=raw_data, x="Timepoint", y="Value", color='gray', alpha=0.3, jitter=True, ax=ax)
    
    # 통계적 유의성 표시
    for i, row in df.iterrows():
        if row["p_value"] is not None:
            label = "유의함 (p<.05)" if row["p_value"] < 0.05 else "유의하지 않음"
            color = "red" if row["p_value"] < 0.05 else "black"
            y_pos = raw_data[raw_data["Timepoint"] == row["Timepoint"]]["Value"].max() + 1.5
            ax.text(i, y_pos, label, ha='center', fontsize=10, color=color, rotation=45)
    
    ax.set_title("예시: 통계적 유의성이 표시된 박스플롯", fontsize=14)
    ax.set_ylabel("측정값")
    ax.set_xlabel("타임포인트")
    ax.set_xticks(range(len(df["Timepoint"])))
    ax.set_xticklabels(df["Timepoint"], rotation=45, ha="right")
    fig.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("""
    ### 이 앱 사용 방법:
    1. t-test 결과 스크린샷을 업로드하세요 (SPSS, Excel 등)
    2. 앱이 OCR을 사용하여 데이터를 추출합니다
    3. 필요한 경우 추출된 데이터를 수동으로 조정할 수 있습니다
    4. 시각화된 결과를 확인하고 다운로드하세요
    
    ### 더 나은 OCR 결과를 위한 팁:
    - 선명하고 고해상도 스크린샷을 사용하세요
    - 테이블이 잘 정렬되고 보이는지 확인하세요
    - 관련 데이터 테이블만 포함하도록 이미지를 자르세요
    """)
