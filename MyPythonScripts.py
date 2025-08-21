import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import RandomHeight
from tensorflow.keras.layers import RandomWidth
#from keras.models import load_model
#from tensorflow.python.keras.layers import Dense
from function import classify
from PIL import Image
import time
from function import annotated_text, pred_prob
import numpy as np

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
model  = load_model('Model_Data_organic_bin_002.h5', 
                    custom_objects={
                        'RandomHeight': RandomHeight,
                        'RandomWidth': RandomWidth
                        })


# Load_ Class Name
class_names = [1, 2]
class_name2 = ['Anorganik','Organik']
class_name3 = ['Silahkan masukkan ke tempat sampah berwarna kuning','Silahkan masukkan ke tempat sampah berwarna hijau']


# Display Image
if file is not None:

    image = Image.open(file)
    st.image(image, use_container_width=True)
    progress_bar = st.progress(0)
    for perc_completed in range(100):
        if (perc_completed == 10):
            st.markdown(" ###### Analyzing & Processing")
        time.sleep(0.02)        
        progress_bar.progress(perc_completed+1)

    # Classify image
    class_name, index, prob = pred_prob(image, model, class_names)

    st.markdown(" ###### Process Completed !")

    with st.expander('Click for prediction Result :'):
         metrics = st.metric(label="Prediction", value=format(class_name2[index]), delta=format(prob))
         st.write("## Keterangan : {} ".format(class_name3[index]))
         #st.write("## Prediction Prob : {:.0%} ".format(prob, '.0%'))
         #st.write("## Action Recommendation : {} ".format(class_name3[index]))





