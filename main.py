import cv2 as cv
import numpy as np

# Open the webcam for capture
cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL
# Read class names from the coco.names file
classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().strip().split('\n')
print(classNames)

# Load model configuration and weights
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Function to find and draw bounding boxes around detected objects
def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    # Process detection outputs
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            
            # Filter detections based on the confidence threshold
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Non-maximum suppression to filter overlapping bounding boxes
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Draw bounding boxes and labels
    for i in np.array(indices).flatten():
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Main loop to process webcam frames
while True:
    success, img = cap.read()
    
    # Preprocess the image and set it as input for the model
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    
    # Get the output layers' names and perform the forward pass
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(outputNames)
    
    # Process the outputs and draw bounding boxes
    findObjects(outputs, img)

    # Show the image with detections
    cv.imshow('Image', img)
    cv.waitKey(1)
