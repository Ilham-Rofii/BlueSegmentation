import streamlit as st
from skimage import io,measure
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage.color import label2rgb
import pandas as pd

st.title("Blue Color Segmentation :red[by Ilham Rofii]")
menu=["Home","About"]
choice=st.sidebar.selectbox("Menu",menu)

def hsv_convert(img):
  hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  mask=cv2.inRange(hsv,(100,90,90),(125,255,255))
  return mask
def binary_closing(mask):
  closed_mask=nd.binary_closing(mask, np.ones((7,7)))
  return(closed_mask)
def label_image(closed_mask):
  label_image=measure.label(closed_mask)
  return label_image
def label_rgb(label_image,img):
  image_label_overlay = label2rgb(label_image,image=img)
  return(image_label_overlay)
def df_extract(label_image,img):
  props=measure.regionprops_table(label_image, img,properties = ['label',
                                                                  'area', 'equivalent_diameter',
                                                                  'mean_intensity','solidity'])
  df=pd.DataFrame(props)
  return df
if choice == "Home":
  st.subheader("Home")
  st.text("Max image size is 100KB due to web hosting limitation")
  image_file=st.file_uploader("Upload Image",type=['PNG','JPG','JPEG'])
  if image_file is not None:
    img = io.imread(image_file)
    img0 = io.imread(image_file)
    st.header("Initial Image")
    st.image(img)
    st.text("")
    img = hsv_convert(img)
    img = binary_closing(img)
    img = label_image(img)
    df = df_extract(img,img0)
    img = label_rgb(img,img0)
    st.header("Labeled Blue Segmentation Image")
    st.image(img)
    st.text("")
    st.header("Table of Labeled Blue Segmentation Image")
    st.write(df)

else:
  st.subheader("About")
  st.text("This program is used to select every part of the image that is blue or close to it.")
