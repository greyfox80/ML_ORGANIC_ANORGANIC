import streamlit as st
import tensorflow as tf
from function import classify
from PIL import Image
import time
import os
from PIL import Image
from annotated_text import annotated_text

# Title
st.title("Selamat Datang di Aplikasi Sorta - Sorting Trash Assistant")

# Header
st.header("Klasifikasi Sampah di Lingkungan Anda")

# Upload the File
file = st.file_uploader(
    "Unggah foto sampah Anda pada kolom unggah di bawah, dan kami akan memprediksi apakah sampah tersebut termasuk organik atau anorganik.",
    type=['jpeg','jpg','png']
)

col1, col2, col3 = st.columns(3)    
with col1:
    pass
with col2:
    pass
with col3:
    annotated_text(
        ("by", "Shafira dan Murin", "#8ef"), 
        ("", "SMAN 12 JKT", "#faa")
    )

# Load Model Classification
custom_objects = {
    "RandomHeight": tf.keras.layers.RandomHeight,
    "RandomFlip": tf.keras.layers.RandomFlip,
    "RandomRotation": tf.keras.layers.RandomRotation,
}
model = tf.keras.models.load_model(
    "Model_Data_organic_002.keras",
    custom_objects=custom_objects
)
#model = tf.keras.models.load_model('Model_Data_organic_002.h5')








