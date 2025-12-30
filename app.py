import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="TB Detection AI", layout="centered")

@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("resnet50_tb_best.keras")
    return model

def home():
    st.title("Welcome to TB Auto Detection")
    st.subheader("This is powered by Deep Learning")
    st.image("D:/Guvi/Projects/TB Detection/Images/Gemini_Generated_Image_nj7p5mnj7p5mnj7p.png")
    
def data_analysis():
    st.header("📊 Model Performance & Data Analysi")

    #1:
    st.subheader("ResNet50 Final Results")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric(label="Accuracy", value="99.56%", delta="Top Performer")
    m2.metric(label="Precision", value="100%", delta="No False Positives")
    m3.metric(label="Recall", value="99.47%", delta="High Sensitivity")
    m4.metric(label="F1-Score", value="0.997", delta="Balanced")

    st.markdown("-----")

    #2:
    st.subheader("Data Insights")
    
    #loading images:
    image1 = Image.open("D:/Guvi/Projects/TB Detection/Images/class.png")
    image2= Image.open("D:/Guvi/Projects/TB Detection/Images/image_dist.png")
    image3= Image.open("D:/Guvi/Projects/TB Detection/Images/pixel.png")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Imbalance")
        st.image(image1, use_container_width=True)
    with col2:
        st.subheader("Images Distribution")
        st.image(image2, use_container_width=True)
    
    st.subheader("Pixel Intensity")
    st.image(image3, use_container_width=True)

    
def detection():
    st.title("🫁 Tuberculosis Detection System")
    st.write("Upload a Chest X-ray to check for signs of Tuberculosis.")
    model  = load_my_model()

    upload_file = st.file_uploader("Upload a Chest X-ray...", type=["jpg", "png", "jpeg"])
    if upload_file is not None:
        image_Xray = Image.open(upload_file).convert('RGB')
        st.image(image_Xray, use_container_width=True)

        with st.spinner('Analyzing the X-ray.....'):
            img= image_Xray.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            #predict 
            prediction = model.predict(img_array)
            prob = prediction[0][0]

            #result:
            st.divider()
            # Create a big bold status box
    if prob > 0.8:
        st.markdown(f"<h1 style='text-align: center; color: white; background-color: #ff4b4b; padding: 20px; border-radius: 10px;'>CRITICAL: TB DETECTED ({prob:.1%})</h1>", unsafe_allow_html=True)
    elif prob > 0.4:
        st.markdown(f"<h1 style='text-align: center; color: black; background-color: #ffeb3b; padding: 20px; border-radius: 10px;'>WARNING: BORDERLINE ({prob:.1%})</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: white; background-color: #28a745; padding: 20px; border-radius: 10px;'>CLEAR: NORMAL ({(1-prob):.1%})</h1>", unsafe_allow_html=True)

pages = {
    "Home": home,
    "EDA" : data_analysis,
    "TB detect" : detection
}

selection = st.sidebar.radio("Choose a page :", list(pages.keys()))
if selection:
    pages[selection]()