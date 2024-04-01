from ultralytics import YOLO
import os
import cv2

output_dir="C:/Users/PRANSHU CHOUBEY/Desktop/Ratneshwar/CODEBUGGED_PREDICT_MASK"
model_path = 'C:/Users/PRANSHU CHOUBEY/Desktop/Ratneshwar/runs/segment/train/weights/best.pt'

image_path = 'C:/Users/PRANSHU CHOUBEY/Desktop/Ratneshwar/abc.jpeg'
filename = os.path.basename(image_path)
img = cv2.imread(image_path)
H, W, _ = img.shape
# CODEBUGGED_TEST\CODEBUGGED_TEST\N8276C2_img_05.jpg
model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))
        file_img=output_dir+"/"+filename
        cv2.imwrite(file_img, mask)