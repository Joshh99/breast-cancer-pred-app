import streamlit as st

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Create a container for title, subtitle
    with st.container():
        st.title("Breast Cancer Prediction App")
        st.write("This app predicts the likelihood of breast cancer using a machine learning model.")

    # Create two columns for the features and predictions
    col1, col2 = st.columns([4, 1])   # 4:1 ratio
    
    with col1:
        st.write("## Features")
    
    with col2:
        st.write("## Prediction")
        
    # Create a sidebar
    st.sidebar.title("Features")
        
if __name__ == "__main__":
    main()