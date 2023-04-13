# Object-Detection
Python script that utilizes the OpenCV library to implement a real-time object detection system using the YOLOv3 (You Only Look Once version 3) algorithm. 
The script captures video from the webcam, processes the frames using the YOLOv3 model, 
and displays the resulting video with bounding boxes around detected objects and their corresponding class labels.

Description of the main sections of the code:

1.Import necessary libraries: The script imports OpenCV (cv2) and NumPy libraries.

2.Initialize video capture and model parameters: It opens the webcam for capturing video and initializes important parameters like the confidence threshold, 
non-maximum suppression (NMS) threshold, and input image size.

3.Load model and class names: The script reads the class names from the "coco.names" file and loads the YOLOv3 model configuration and weights.

4.Define the findObjects function: This function processes the output of the YOLOv3 model and draws bounding boxes and labels around detected objects. 
It filters the detections based on the confidence threshold and applies non-maximum suppression to eliminate overlapping boxes.

5.Main loop: The script enters an infinite loop to process webcam frames in real-time. In each iteration, it captures a frame, preprocesses the image, 
performs a forward pass through the YOLOv3 model, processes the output using the findObjects function, and displays the resulting frame with bounding boxes and labels.


The script runs continuously until the user closes the window displaying the video feed. The real-time object detection can be useful in various applications like video surveillance, 
robotics, or any other scenario where immediate identification of objects in a video stream is essential.






https://user-images.githubusercontent.com/76581527/231832645-b99e9313-2515-4589-a29b-f5b362a93677.mp4

