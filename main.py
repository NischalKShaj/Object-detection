# file to predict the images

# importing the required modules
from ultralytics import YOLO 
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt


# loading the trained model
model = YOLO("predict.pt")

# directory containing test image
fruit_dir = "fruit_dir"
output_csv = "predict.csv"
output_image_dir = "predicted_images"
os.makedirs(output_image_dir, exist_ok=True)


# storage for prediction
results_data = []

# iterate through images in the folder
for filename in os.listdir(fruit_dir):
    if filename.lower().endswith((".jpeg", ".jpg", ".png")):
        image_path = os.path.join(fruit_dir, filename)
        image = cv2.imread(image_path)

        # perform inference
        results = model(image_path, conf=0.7)

        # if predictions found
        for r in results:
            if len(r.boxes.cls) > 0:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(r.boxes.cls[0])
                    confidence = float(r.boxes.conf[0])
                    class_name = model.names[class_id]

                    # for drawing rect and labeling
                    label = f"{class_name} {confidence:.2f}"
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

                    results_data.append({
                        "Filename":filename,
                        "Predicted Class":class_name,
                        "Confidence":round(confidence, 4)
                    })
            else:
                results_data.append({
                    "Filename":filename,
                    "Predicted Class": class_name,
                    "Confidence": 0.0
                })

        # saving the annotated image
        output_image_path = os.path.join(output_image_dir, filename)
        cv2.imwrite(output_image_path, image)                


# save prediction to csv
df = pd.DataFrame(results_data)
df.to_csv(output_csv, index=False)

print(f"Prediction saved to :{output_csv}")
print(f"Annotated images saved to: {output_image_dir}/")