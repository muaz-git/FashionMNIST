from imageai.Detection import ObjectDetection
import os
import cv2
import json

video_path = '/path/to/video'
model_path = os.path.join("/path/to/model", "resnet50_coco_best_v2.0.1.h5")
output_path = 'file.txt'

video = cv2.VideoCapture(video_path)

execution_path = os.getcwd()
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

fn = 0
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

complete_dict = {}
while True:
    print(fn)
    ret, frame = video.read()
    if not ret:
        break

    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = detector.detectObjectsFromImage(input_image=cv2_frame,
                                                 output_image_path=os.path.join(execution_path, "imagenew.png"),
                                                 input_type="array",
                                                 minimum_percentage_probability=1)

    objs = []
    for eachObject in detections:

        my_arr = []
        for elem in eachObject["box_points"]:
            my_arr.append(elem.item())

        objs.append(my_arr)
    complete_dict[str(fn)] = objs

    fn += 1

with open(output_path, 'w') as file:
    file.write(json.dumps(complete_dict))

detector = ObjectDetection()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.png"),
                                             output_image_path=os.path.join(execution_path, "imagenew.png"),
                                             minimum_percentage_probability=1)
