# main file for the object detection project

# importing the required modules
import cv2
import csv
from datetime import datetime
from ultralytics import YOLO 

# loading the model
model = YOLO("yolov8n.pt")

# setting up the allowed classes
allowed_classes = ["person", "chair", "book"]

# get the names from the models
model_classes = model.names

# open the default webcam
cap = cv2.VideoCapture(0)

# setting up the csv logging    
csv_file = open("detection_report.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Class", "Confidence"])

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # for adding auto brightness
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)

    # performs declaration
    result = model.predict(source=frame, show=False)
 
    if result and len(result)>0:
        boxes = result[0].boxes
        #plot the result on the frame
        annotated_frame = frame.copy()

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model_classes[cls_id]

            if class_name in allowed_classes:
                xyxy =box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].item()

                # draw rectangle and label
                cv2.rectangle(annotated_frame,(xyxy[0],xyxy[1]),(xyxy[2],xyxy[3]),(0,255,0),2)
                cv2.putText(annotated_frame,f"{class_name} {conf:2f}",(xyxy[0],xyxy[1] -10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)

                
                # log detection to csv
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                csv_writer.writerow([timestamp, class_name,conf])


        # display in window
        cv2.imshow("Filered Webcam Detection", annotated_frame)
    else:
        cv2.imshow("Filtered Webcam Detection",frame)


    # exit on pressing q
    if cv2.waitKey(1000) & 0xFF == ord("q"):
        break

# release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
csv_file.close()