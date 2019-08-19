from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os
import cv2
import numpy as np
import json

video_path = '/home/mumu01/Projects/sample_video.mkv'
video = cv2.VideoCapture(video_path)

# New cv2
execution_path = os.getcwd()
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

fn = 0
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join("/home/mumu01/Projects/models", "resnet50_coco_best_v2.0.1.h5"))
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
        # print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
        my_arr = []
        for elem in eachObject["box_points"]:
            my_arr.append(elem.item())

        objs.append(my_arr)
    complete_dict[str(fn)] = objs
    # break
    fn += 1

output_path = 'file.txt'
with open('file.txt', 'w') as file:
    file.write(json.dumps(complete_dict))
# # from imageai.Prediction.Custom import ModelTraining
# #
# # model_trainer = ModelTraining()
# # model_trainer.setModelTypeAsResNet()
# # model_trainer.setDataDirectory("data/FashionMNIST/original")
# # model_trainer.trainModel(num_objects=10, num_experiments=200, enhance_data=True, batch_size=32, show_network_summary=True)
#
#
# execution_path = os.getcwd()
# # detector = VideoObjectDetection()
detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath(os.path.join("/home/mumu01/Projects/models", "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "image.png"),
                                             output_image_path=os.path.join(execution_path, "imagenew.png"),
                                             minimum_percentage_probability=1)
#
# for eachObject in detections:
#     print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
#
# exit()
