import streamlit as st
import pickle
import tensorflow as tf
#from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.models import load_model
#from tensorflow.python.keras.layers import Dense
from function import classify
from PIL import Image
import time
from annotated_text import annotated_text

# Title
st.title("Selamat Datang di Aplikasi Sorta - Sorting Trash Assistant")

# Header
st.header("Klasifikasi Sampah di Linkungan Anda")

# Upload the File
file = st.file_uploader("Unggah foto sampah Anda pada kolom unggah di bawah, dan kami akan memprediksi apakah sampah tersebut termasuk organik atau anorganik.", type=['jpeg','jpg','png'])

col1, col2, col3 = st.columns(3)    
with col1:
    """
    """
with col2:
    """
    """    
with col3:
    annotated_text(
                ("by","Shafira dan Murin","#8ef"),("","SMAN 12 JKT","#faa")
              )

# Load Model Classification
model  = tf.keras.models.load_model('Model_Data_organic_002.keras')














