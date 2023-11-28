import os

from tempfile import NamedTemporaryFile
import cv2
from tqdm import tqdm
from efficientnet.tfkeras import EfficientNetB3
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf


def file_selector(folder_path='.'):
    """
    Helper function for streamlit selector
    :param folder_path: Path to folder
    :return: File name
    """
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def decode_image(filename, image_size=(512, 512)):
    """
    Decode image for model.h5 as correct input
    :param filename: File name
    :param image_size: Size of image (default 512x512x3)
    :return: Decoded image for model input
    """
    bits = tf.io.read_file(filename)
    image1 = tf.image.decode_jpeg(bits, channels=3)
    image1 = tf.cast(image1, tf.float32) / 255.0
    image1 = tf.expand_dims(image1, axis=0)
    image1 = tf.image.resize(image1, image_size)

    return image1


def main():
    """
    Main streamlit run file
    :return: Application
    """
    st.set_page_config(layout="wide", page_title="Steganography Detect by Deep learning")
    st.title("Steganography Detect by Deep learning Web App")
    st.sidebar.title("Steganography Detect using deep learning Web App")
    st.sidebar.caption("Do you know that steganography can be hidden everywhere, even in the most innocuous-looking"
                       " digital images? Our steganography detection app is designed to unveil concealed messages"
                       " and content in images, ensuring your data remains secure. Whether you're concerned about"
                       " potential threats or simply curious about the hidden world of digital communication,"
                       " our app empowers you to reveal what lies beneath the surface."
                       " With advanced algorithms and intuitive user interfaces,"
                       " we make steganalysis accessible to all, allowing you to protect your privacy"
                       " and stay one step ahead of potential data breaches.")
    st.markdown("Is your image contains steganography?")
    st.sidebar.markdown("Do you want to know that your image contains steganography?")
    model = tf.keras.models.load_model("models/main_model.h5")

    uploaded_file = st.file_uploader("Choose a image file", type="jpg")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, width=200, channels='BGR')

    if st.button("Classify", key='classify1'):
        with NamedTemporaryFile(dir='.', suffix='.csv') as f:
            f.write(uploaded_file.getbuffer())
            image = decode_image(f.name)
            predict = model.predict(image, verbose=1)
            predicted = "{:.2f}".format(predict[0][0])
            # format to two decimal places
            float_value = float(predicted)
            print(float_value)
            print(f"Presence of steganography is near: {float_value}")
            st.write(f"Presence of steganography is near: {float_value}")


if __name__ == '__main__':
    main()
