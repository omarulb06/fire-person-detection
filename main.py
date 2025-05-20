import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the classification model and processor for fire detection
fire_processor = AutoImageProcessor.from_pretrained("EdBianchi/vit-fire-detection", use_fast=True)
fire_model = AutoModelForImageClassification.from_pretrained("EdBianchi/vit-fire-detection")

# Load YOLO model for person detection
yolo_net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load COCO names for YOLO class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Open a connection to the webcam
cap = cv2.VideoCapture(0)  # Use the appropriate device index if you have multiple webcams

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Person Detection Section (YOLO)
    # Prepare the image for the YOLO model
    yolo_blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(yolo_blob)
    yolo_outs = yolo_net.forward(output_layers)

    # YOLO detection information to display
    yolo_class_ids = []
    yolo_confidences = []
    yolo_boxes = []

    # Loop through the YOLO outputs
    for out in yolo_outs:
        for detection in out:
            yolo_scores = detection[5:]
            yolo_class_id = np.argmax(yolo_scores)
            yolo_confidence = yolo_scores[yolo_class_id]
            
            # Filter for people (class_id == 0 in COCO dataset)
            if classes[yolo_class_id] == 'person' and yolo_confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates for the person
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                yolo_boxes.append([x, y, w, h])
                yolo_confidences.append(float(yolo_confidence))
                yolo_class_ids.append(yolo_class_id)

    # Apply Non-Maximum Suppression to remove redundant boxes
    yolo_indexes = cv2.dnn.NMSBoxes(yolo_boxes, yolo_confidences, 0.5, 0.4)

    # If at least one person is detected, perform fire detection
    if len(yolo_indexes) > 0:
        # Draw bounding boxes on the detected people
        for i in yolo_indexes.flatten():
            x, y, w, h = yolo_boxes[i]
            label = str(classes[yolo_class_ids[i]])
            confidence = yolo_confidences[i]
            color = (0, 255, 0)  # Green color for person detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Fire Detection Section
        # Convert the frame to PIL Image for the fire detection model
        fire_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image for the fire detection model
        fire_inputs = fire_processor(images=fire_image, return_tensors="pt")

        # Make fire detection prediction
        with torch.no_grad():
            fire_outputs = fire_model(**fire_inputs)

        # Get predicted class for fire detection
        fire_logits = fire_outputs.logits
        fire_predicted_class_idx = fire_logits.argmax(-1).item()
        
        # Fire detection class labels
        fire_labels = ["Fire", "No Fire"]

        # Show message if fire is detected
        if fire_labels[fire_predicted_class_idx] == "Fire":
            fire_message = "Fire detected!"
            cv2.putText(frame, fire_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            fire_message = "No Fire"
            cv2.putText(frame, fire_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # If no person is detected, show "No person detected" message
    else:
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Fire and Person Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
