import  streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import os

st.title("YOLO Image and Vedio Preprocessing ")
upload_file=st.file_uploader("upload an image or vedio file",type=['jpg','png','jpeg','mp4','mkv'])
model=YOLO("best.pt")
video_holder=st.empty()

def process_media(input_path,output_path): 
    file_extension = os.path.splitext(input_path)[1].lower() 
    if file_extension in ['.mp4', '.mkv']:
        return predict_video(input_path,output_path)
    elif file_extension in ['.jpg','.jpeg','.png']:
        return predict_and_save_image(input_path,output_path)
    else:
        st.error(f"unsupported file type:{file_extension}")
        return None


def predict_and_save_image(path_test_car,output_image_path):
    result=model.predict(path_test_car,device='cpu')
    image=cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for result in result: 
        for box in result.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0]) 
            confidence=box.conf[0]
            cv2.rectangle(image, (x1, y1),(x2, y2), (0, 255, 0), 2)
            cv2.putText(image,f'{confidence*100:.2f}%',
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path,image)
        return output_image_path 
    
def predict_video(input_video_path,output_image_path):
    cam=cv2.VideoCapture(input_video_path)
    model=YOLO("best.pt")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(upload_file.name.replace(".mp4",".avi"), fourcc, 20.0, (frame_width, frame_height))
    while True:
        has_frame,frame=cam.read()
        results=model.predict(frame)
        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2=map(int,box.xyxy[0])
                conf=box.conf[0]
                cv2.rectangle(frame,(x1,y1),(x2,y2),(225,0,0),2)
                cv2.putText(frame,f'{conf*100:.2f}%',
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
            out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_holder.image(frame_rgb, channels="RGB")
    
if upload_file is not None:
    input_path = f"temp/{upload_file.name}"
    output_path = f"output/{upload_file.name}"
    with open(input_path,'wb') as f:
        f.write(upload_file.getbuffer())
        st.write("preprocessing Image.....")
        result_path=process_media(input_path, output_path)
        ## build logic for prediction
        if result_path:
            if input_path.endswith(('.mp4','mkv')):
                st.video(upload_file.name.replace(".mp4",".avi"))
            else:
                st.image(result_path)