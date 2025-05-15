
# streamlit_ttest_plot.py

import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

st.set_page_config(layout="wide")
st.title("Paired t-test Boxplot Visualizer from Screenshot")

st.markdown("""
This app allows you to:
1. Upload a screenshot image (e.g., SPSS output or Excel screenshot).
2. Extract numerical values using OCR.
3. Simulate data using mean, std, and N.
4. Visualize boxplots per timepoint with significance annotations.
""")

uploaded_file = st.file_uploader("Upload your t-test result screenshot (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Step 1: OCR Extracted Text")
    text = pytesseract.image_to_string(image, lang='eng+kor')
    st.text(text)

    st.subheader("Step 2: Example Simulated Table")

    # Sample manually entered data based on your image
    data = {
        "Timepoint": ["NB", "NA1", "NA2", "NA3", "NA4", "NA5", "NFU"],
        "Mean": [104.6667, 105.2286, 105.1810, 104.6667, 102.8571, 102.9810, 106.1692],
        "SD": [6.7413, 6.4390, 6.1071, 6.7413, 6.9111, 6.9561, 6.2758],
        "N": [42, 42, 42, 42, 42, 42, 26],
        "p_value": [None, 0.024, 0.106, 0.008, 0.001, 0.000, 0.004]
    }
    df = pd.DataFrame(data)
    st.dataframe(df)

    st.subheader("Step 3: Simulated Raw Data for Boxplot")
    raw_data = pd.DataFrame([
        (row["Timepoint"], np.random.normal(row["Mean"], row["SD"], row["N"]))
        for _, row in df.iterrows()
    ], columns=["Timepoint", "Value"]).explode("Value")
    raw_data["Value"] = raw_data["Value"].astype(float)
    
    st.write(raw_data.head())

    st.subheader("Step 4: Boxplot Visualization")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=raw_data, x="Timepoint", y="Value", palette="pastel")
    sns.stripplot(data=raw_data, x="Timepoint", y="Value", color='gray', alpha=0.3, jitter=True)

    for i, row in df.iterrows():
        if row["p_value"] is not None:
            label = "Significant (p<.05)" if row["p_value"] < 0.05 else "Not significant"
            y_pos = raw_data[raw_data["Timepoint"] == row["Timepoint"]]["Value"].max() + 1.5
            plt.text(i, y_pos, label, ha='center', fontsize=10, color="black")

    plt.title("Boxplot per Timepoint with Statistical Significance", fontsize=14)
    plt.ylabel("Measurement Value")
    plt.xlabel("Timepoint")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.success("Visualization Complete!")
else:
    st.info("Please upload a screenshot to begin.")
