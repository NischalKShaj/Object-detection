# file to predict the images

# importing the required modules
from ultralytics import YOLO 
import os
import pandas as pd
import cv2
import mathpoltlib.pyplot as plt


# loading the trained model
model = YOLO("best.pt")

# directory containing test image
fruit_dir = "fruit_dir"
output_csv = "prediction.csv"


# storage for prediction
results_data = []

# iterate through images in the folder
for filename in os.listdir(fruit_dir):
    if filename.lower().endswith((".jpeg", ".jpg", ".png")):
        image_path = os.path.join(image_folder, filename)

        # perform inference
        results = model(image_path, conf=0.3)

        # if predictions found
        for r in results:
            if len(r.boxes.cls)>0:
                class_id = int(r.boxes.cls[0])
                confidence = float(r.boxes.conf[0])
                class_name = model.names[class_id]
            else:
                class_name = "No Detection"
                confidence = 0.0 

            # append the data
            results_data.append({
                "Filename":filename,
                "Predicted Class":class_name,
                "Confidence":round(confidence, 4)
            })


# save prediction to csv
df = pd.DataFrame(results_data)
df.to.csv(output_csv, index=False)

print(f"Prediction saved to :{output_csv}")