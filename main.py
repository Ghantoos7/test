import streamlit as st
from src.analytics.visualization import analyze_order_spectrograms, order_spectro_plot_plotly
import pandas as pd
import os
from src.model.prediction import predict
from src.preprocessing.preprocess import preprocess_csv
import time

def main():
    st.set_page_config(page_title="Order Spectrogram Analysis", layout="wide")

    # Center the main title
    st.markdown(
        "<h1 style='text-align: center;'>Körperschallprüfung Analyzer</h1>",
        unsafe_allow_html=True
    )

    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    # Sidebar for file upload
    with st.sidebar:
        st.header("File Upload")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Display a loading bar while saving and preprocessing the file
        with st.spinner('Uploading and preprocessing file...'):
            raw_file_path = os.path.join(raw_data_path, uploaded_file.name)
            os.makedirs(os.path.dirname(raw_file_path), exist_ok=True)
            with open(raw_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            processed_file_path = preprocess_csv(raw_file_path)
            processed_file_name = os.path.basename(processed_file_path)
            final_processed_file_path = os.path.join(processed_data_path, processed_file_name)
            os.makedirs(os.path.dirname(final_processed_file_path), exist_ok=True)
            os.rename(processed_file_path, final_processed_file_path)
        st.success('File uploaded and preprocessed successfully!')

    # Select a file from processed files
    st.sidebar.header("Select a Dataset")
    if os.path.exists(processed_data_path):
        file_list = os.listdir(processed_data_path)
        selected_file = st.sidebar.selectbox('Select a preprocessed CSV file', file_list)

        if selected_file:
            file_path = os.path.join(processed_data_path, selected_file)
            df = pd.read_csv(file_path)
            name = os.path.basename(file_path)
            machine_name = name.split("\\")[-1].split("_")[0]

            # Button to run model prediction
            if st.sidebar.button("Run Model and Analyze"):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                progress_text.write("Starting model prediction and analysis...")

                # Simulate a progress bar
                for percent_complete in range(1, 101):
                    time.sleep(0.07)  # Simulate work being done
                    progress_bar.progress(percent_complete)
                    progress_text.write(f"Progress: {percent_complete}%")
                # Hide progress bar and text after completion
                progress_bar.empty()
                progress_text.empty()

                result = predict(file_path)

                # Display the model result
                if result == 0:
                    st.markdown(f"<h1 style='text-align: center; color: green;'>{machine_name} is OKAY</h1>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h1 style='text-align: center; color: red;'>{machine_name} is NOT OKAY</h1>", unsafe_allow_html=True)

                # Generate visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.spinner('Generating order spectrogram analysis...'):
                        order_spectrogram_fig = analyze_order_spectrograms(df, machine_name)
                        st.plotly_chart(order_spectrogram_fig, use_container_width=True)

                with col2:
                    with st.spinner('Generating acceleration heatmap...'):
                        acceleration_heatmap = order_spectro_plot_plotly(df, machine_name)
                        st.plotly_chart(acceleration_heatmap, use_container_width=True)

                # Display the dataset
                st.subheader("Dataset")
                st.dataframe(df)

if __name__ == "__main__":
    main()