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
#model = tf.keras.models.load_model('Model_Data_organic_002.keras')

# Custom loss function to handle deserialization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class CustomSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):
    def __init__(self, reduction='auto', name='sparse_categorical_crossentropy', from_logits=False, ignore_class=None):
        super().__init__(reduction=reduction, name=name, from_logits=from_logits, ignore_class=ignore_class)

# Register the custom loss function
tf.keras.utils.get_custom_objects().update({
    'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy
})

# Set working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'Model_Data_organic_002.h5')

# Load the pre-trained model
model = tf.keras.models.load_model(model_path, custom_objects={'SparseCategoricalCrossentropy': CustomSparseCategoricalCrossentropy})




