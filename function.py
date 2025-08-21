import base64
import streamlit as st
import pickle
import tensorflow as tf
from PIL import ImageOps, Image
import numpy as np

# Create a function to load and prepare image
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape, color_channels)

  Args :
    filename (str) : target image filepath
    img_shape (int) : height/widht dimension or size to resize target image to, default 224
    scale (bool) : whether to scale pixel values from 0-255 to range(0, 1), default True
  """
  # Read in the image
  img = tf.convert_to_tensor(filename)
  #img = tf.io.read_file(file)
  # Decode the read file into a tensor & ensure 3 color channels
  #img = tf.image.decode_image(img, channels=3)
  # Resize the image
  img = tf.image.resize(img, size=[img_shape, img_shape])

  # Scale ? yes/No
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img


def classify(image, model, class_names):
    # Convert image to (224,224)
    img  = load_and_prep_image(image, scale=False)
    image_expanded = tf.expand_dims(img, axis=0)
    
    # Make Prediction
    pred_prob = model.predict(image_expanded)
    index = pred_prob.argmax() #np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = index
    predictions_prob = pred_prob.max()

    return class_name, confidence_score, predictions_prob

def pred_prob(image, model, class_names):
    # Convert image to (224,224)
    img  = load_and_prep_image(image, scale=False)
    image_expanded = tf.expand_dims(img, axis=0)
    
    # Make Prediction
    pred_prob = model.predict(image_expanded)
    index = int(tf.round(pred_prob))
    class_name = class_names[index]
    confidence_score = index
    predictions_prob = pred_prob.max()

    return class_name, confidence_score, predictions_prob

def annotated_text(*args):
    """Display text with annotations in Streamlit."""
    out = ""
    for arg in args:
        if isinstance(arg, str):
            out += arg
        elif isinstance(arg, tuple):
            text, label, color = arg
            out += f'<span style="background-color:{color};border-radius:0.2em;padding:0.2em 0.4em;margin:0 0.1em;">{text} <span style="opacity:0.7;font-size:0.8em;">{label}</span></span>'
    st.markdown(out, unsafe_allow_html=True)






