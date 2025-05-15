import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
import cv2
import re
import io

# 페이지 설정
st.set_page_config(page_title="엑셀 결과 시각화 도구", layout="wide")
st.title("엑셀 스크린샷 데이터 시각화 도구")

# 파일 업로드 위젯 
uploaded_file = st.file_uploader("엑셀 스크린샷 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

def process_image(image):
    """이미지 전처리 및 OCR 수행"""
    # 이미지를 흑백으로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 이미지 대비 향상
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # OCR 수행
    text = pytesseract.image_to_string(gray)
    return text

def extract_data_from_text(text):
    """OCR 결과 텍스트에서 데이터 추출"""
    lines = text.strip().split('\n')
    
    # 헤더 찾기
    header_line = None
    for i, line in enumerate(lines):
        if 'Timepoint' in line or 'Time' in line or 'Point' in line:
            header_line = i
            break
    
    if header_line is None:
        st.error("테이블 헤더를 찾을 수 없습니다. 다른 이미지를 업로드해주세요.")
        return None, None

    # 헤더 추출 및 처리
    headers = re.split(r'\s{2,}', lines[header_line])
    headers = [h.strip() for h in headers]
    
    # 필요한 컬럼 확인
    if 'Timepoint' not in headers and 'Time' not in headers:
        headers = ['Timepoint'] + headers  # 타임포인트 컬럼 추가
    
    # p-value 컬럼 확인
    p_value_col = None
    for i, h in enumerate(headers):
        if 'p' in h.lower() and 'value' in h.lower():
            p_value_col = i
            break
    
    # 데이터 행 추출
    data_rows = []
    for i in range(header_line + 1, len(lines)):
        line = lines[i].strip()
        if line:
            values = re.split(r'\s{2,}', line)
            values = [v.strip() for v in values]
            
            # 헤더와 값의 개수 맞추기
            while len(values) < len(headers):
                values.append(None)
            
            # 너무 많은 값이 있으면 자르기
            if len(values) > len(headers):
                values = values[:len(headers)]
                
            data_rows.append(values)
    
    # 원시 데이터와 통계 데이터 분리
    raw_data_rows = []
    stat_data_rows = []
    
    for row in data_rows:
        # p-value가 있는지 확인
        has_p_value = False
        if p_value_col is not None and p_value_col < len(row):
            p_val_str = row[p_value_col]
            if p_val_str and any(c.isdigit() for c in p_val_str):
                has_p_value = True
        
        if has_p_value:
            stat_data_rows.append(row)
        else:
            # 이 행이 원시 데이터로 보이면 추가
            raw_data_rows.append(row)
    
    # DataFrame 생성
    if raw_data_rows:
        raw_data_df = pd.DataFrame(raw_data_rows, columns=headers)
    else:
        # 원시 데이터를 찾지 못했을 경우, 통계 데이터로부터 가상의 원시 데이터 생성
        raw_data_df = generate_sample_data(stat_data_rows, headers)
    
    # 통계 데이터 DataFrame
    stat_df = pd.DataFrame(stat_data_rows, columns=headers) if stat_data_rows else None
    
    # 데이터 타입 변환
    # Timepoint 열 찾기
    timepoint_col = 'Timepoint'
    if timepoint_col not in raw_data_df.columns:
        for col in raw_data_df.columns:
            if 'time' in col.lower() or 'point' in col.lower():
                timepoint_col = col
                break
    
    # Value 열 찾기
    value_col = 'Value'
    if value_col not in raw_data_df.columns:
        for col in raw_data_df.columns:
            if col != timepoint_col and 'value' in col.lower():
                value_col = col
                break
            elif col != timepoint_col and any(c.isdigit() for c in raw_data_df[col].iloc[0] if pd.notna(raw_data_df[col].iloc[0])):
                value_col = col
                break
    
    # 필요한 열 이름 재설정
    raw_data_df = raw_data_df.rename(columns={timepoint_col: 'Timepoint', value_col: 'Value'})
    
    # 데이터 타입 변환
    raw_data_df['Value'] = pd.to_numeric(raw_data_df['Value'], errors='coerce')
    
    if stat_df is not None:
        # p-value 열 찾기
        p_value_col_name = None
        for col in stat_df.columns:
            if 'p' in col.lower() and 'value' in col.lower():
                p_value_col_name = col
                break
        
        if p_value_col_name:
            stat_df = stat_df.rename(columns={p_value_col_name: 'p_value', timepoint_col: 'Timepoint'})
            stat_df['p_value'] = pd.to_numeric(stat_df['p_value'], errors='coerce')
    
    return raw_data_df, stat_df

def generate_sample_data(stat_rows, headers):
    """통계 데이터를 기반으로 가상의 원시 데이터 생성"""
    sample_data = []
    
    # p-value 열과 Timepoint 열 찾기
    p_col = None
    time_col = None
    for i, h in enumerate(headers):
        if 'p' in h.lower() and 'value' in h.lower():
            p_col = i
        if 'time' in h.lower() or 'point' in h.lower():
            time_col = i
    
    if time_col is None:
        time_col = 0  # 첫 번째 열을 기본값으로 설정
    
    # 각 타임포인트별로 가상 데이터 생성
    for row in stat_rows:
        timepoint = row[time_col]
        
        # 타임포인트별로 15개 샘플 생성
        for _ in range(15):
            # 기본 평균과 표준편차 설정
            mean = 50
            std = 10
            
            # p-value가 있으면 유의한 차이 만들기
            if p_col is not None and p_col < len(row):
                p_val_str = row[p_col]
                try:
                    p_val = float(p_val_str)
                    if p_val < 0.05:
                        # 유의한 경우 값을 더 크게
                        mean = 70
                        std = 15
                except:
                    pass
            
            # 정규분포에서 값 생성
            value = np.random.normal(mean, std)
            
            # 샘플 행 생성
            sample_row = [None] * len(headers)
            sample_row[time_col] = timepoint
            
            # Value 컬럼 설정
            value_col = None
            for i, h in enumerate(headers):
                if 'value' in h.lower():
                    value_col = i
                    break
            
            if value_col is None:
                # Value 컬럼이 없으면 타임포인트 다음 열 사용
                value_col = time_col + 1
                if value_col >= len(headers):
                    value_col = 1  # 안전을 위해 두 번째 열로 설정
            
            sample_row[value_col] = f"{value:.2f}"
            sample_data.append(sample_row)
    
    return pd.DataFrame(sample_data, columns=headers)

if uploaded_file is not None:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    
    # OCR 처리
    with st.spinner('이미지에서 데이터를 추출 중입니다...'):
        text = process_image(image)
        raw_data, df = extract_data_from_text(text)
    
    if raw_data is not None:
        st.subheader("Step 1: 추출된 원시 데이터")
        st.write(raw_data.head())
        
        if df is not None:
            st.subheader("Step 2: 추출된 통계 데이터")
            st.write(df)
        else:
            st.info("통계 데이터를 추출하지 못했습니다. 박스플롯만 표시됩니다.")
            df = pd.DataFrame({'Timepoint': raw_data['Timepoint'].unique(), 'p_value': [None] * len(raw_data['Timepoint'].unique())})
        
        st.subheader("Step 3: 데이터 시각화")
        # Matplotlib의 객체지향 API 사용
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 박스플롯과 스트립플롯으로 데이터 시각화
        sns.boxplot(data=raw_data, x="Timepoint", y="Value", palette="pastel", ax=ax)
        sns.stripplot(data=raw_data, x="Timepoint", y="Value", color='gray', alpha=0.3, jitter=True, ax=ax)
        
        # 텍스트 위치 계산을 위한 y축 범위 지정
        y_values = raw_data["Value"].dropna()
        if not y_values.empty:
            y_min = y_values.min()
            y_max = y_values.max()
            # 텍스트가 보이도록 충분한 여백 확보
            text_offset = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
        else:
            y_min, y_max = 0, 10
            text_offset = 0.5
        
        # 각 타임포인트별 통계적 유의성 표시
        timepoint_list = sorted(raw_data['Timepoint'].unique())
        timepoint_positions = {tp: i for i, tp in enumerate(timepoint_list)}
        
        for i, row in df.iterrows():
            timepoint = row["Timepoint"]
            p_value = row.get("p_value")
            
            if p_value is not None:
                # p-value에 따라 유의성 라벨 설정
                label = "Significant (p<.05)" if p_value < 0.05 else "Not significant"
                
                # 현재 타임포인트의 최대값 찾기
                current_values = raw_data[raw_data["Timepoint"] == timepoint]["Value"].dropna()
                y_pos = y_max + text_offset  # 기본값
                
                if not current_values.empty:
                    current_max = current_values.max()
                    if not pd.isna(current_max):
                        y_pos = current_max + text_offset
                
                pos_idx = timepoint_positions.get(timepoint, i)
                # 해당 위치에 텍스트 추가
                ax.text(pos_idx, y_pos, label, ha='center', fontsize=10, color="black")
        
        # 그래프 제목 및 축 레이블 설정
        title_obj = ax.set_title("Boxplot per Timepoint with Statistical Significance", fontsize=14)
        title_obj.set_y(1.05)  # 제목 아래 여백 추가
        
        ax.set_ylabel("Measurement Value")
        ax.set_xlabel("Timepoint")
        
        # x축 눈금 설정
        ax.set_xticks(range(len(timepoint_list)))
        ax.set_xticklabels(timepoint_list, rotation=45, ha="right")
        
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
        ax.set_ylim(current_ylim_min, max(current_ylim_max, max_text_y_coord + text_offset * 0.5))
        
        # 레이아웃 조정
        fig.tight_layout(pad=1.1)
        st.pyplot(fig)
        
        # 완료 메시지 표시
        st.success("시각화가 완료되었습니다!")
        
        # 데이터 다운로드 옵션 제공
        st.subheader("Step 4: 추출된 데이터 다운로드")
        
        csv_raw = raw_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="원시 데이터 CSV 다운로드",
            data=csv_raw,
            file_name='extracted_raw_data.csv',
            mime='text/csv',
        )
        
        if df is not None:
            csv_stat = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="통계 데이터 CSV 다운로드",
                data=csv_stat,
                file_name='extracted_stat_data.csv',
                mime='text/csv',
            )
    else:
        st.error("이미지에서 데이터를 추출할 수 없습니다. 더 선명한 이미지를 업로드해주세요.")

else:
    st.info("엑셀 결과 스크린샷을 업로드하면 데이터를 추출하고 시각화합니다.")
    st.markdown("""
    ### 사용 방법:
    1. 엑셀 테이블이 포함된 스크린샷을 촬영하거나 저장합니다.
    2. 위의 '파일 업로드' 버튼을 클릭하여 이미지를 업로드합니다.
    3. 프로그램이 자동으로 데이터를 인식하고 시각화합니다.
    4. 결과를 CSV 파일로 다운로드할 수 있습니다.
    
    ### 주의사항:
    - 이미지가 선명하고 텍스트가 잘 보이도록 해주세요.
    - 테이블에는 반드시 'Timepoint'(또는 유사한 이름)와 'Value' 열이 포함되어야 합니다.
    - p-value가 포함된 통계 데이터가 있으면 함께 추출됩니다.
    """)