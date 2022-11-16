import streamlit as st
import pickle
from PIL import Image
import os
import cv2
import numpy as np
import pandas as pd

model_dir = r"C:\Users\SAI\Desktop\Traffic Sign Detectiona and Recognition\trained models\pedestrian"

knn_model = pickle.load(open(os.path.join(model_dir, "model_knn.sav"), "rb"))

def welcome():
    return "Welcome to Traffic Sign Detection and Recognition"

def processImg(img_path):
    print(img_path)
    # i = cv2.imread(img_path)
    # cv2.imshow("Image", i)
    # cv2.waitKey(0) 
  
    # #closing all open windows 
    # cv2.destroyAllWindows() 
    try:
        path = os.path.join(img_path)
        a = cv2.imread(path)
        resize = (280,430)
        img = cv2.resize(a, resize)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        out = pd.DataFrame(descriptors)
        
        try:
            array_double = np.array(out, dtype=np.double)
            a = knn_model.predict(array_double)
            hist=np.histogram(a,bins=[0,1,2,3,4,5])
            print("HISTOGRAM",hist)
            return a
        except ValueError as v:
            print("Something went wrong..!")
            print("Error: ", v)
            return "Error"
        
    except Exception as e:
        print(str(e))
        return "Error"

def main(img_path):
    img = processImg(img_path).tolist()
    # print(img)
    return pd.DataFrame([
        ["Pedestrian", img.count(1)/(img.count(0)+img.count(1))*100],
        ["Other", img.count(0)/(img.count(0)+img.count(1))*100]
    ], columns=["Sign", "Accuracy"])
    # if (img.count(0) > img.count(1)):
    #     return "Not a Pedestrian"
    # else:
    #     return ("Pedestrian: ", img.count(0)/(img.count(0)+img.count(1))*100, "%")

if __name__ == '__main__':
    st.title("Traffic Sign Detection and Recognition")
    # img_path = st.text_input("Enter the image path")
    file = st.file_uploader("Upload the image", type=["jpg", "png"], label_visibility="collapsed")
    # if st.button("Submit"):
    #     result = main(img_path)
    #     st.table(result) 

    if file is not None:
        st.write(file.name)
        data = file.getvalue()

        with open(f"./uploaded/{file.name}", "wb") as f:
            f.write(data)
        
        result = main(f"./uploaded/{file.name}")
        st.table(result) 
    
    else:
        st.write("No file uploaded")
