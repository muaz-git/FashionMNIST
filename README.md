##### Object deteciton in video
To detect objects in a video, first generate some proposals using [imageai](https://github.com/OlafenwaMoses/ImageAI). Follow [this tutorial](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606) to setup imageai.

Note that choice of imageai is arbitrary and any other object detector library can be used. 
###### Proposal generator for the video
To get proposals run `python proposal_generator.py`. Make sure to set correct `video_path`. A dictionary of all the bounding box (BB) proposals will be saved at the location `output_path`
###### Testing on video
Once BB proposals have been generated. Run `python test_video.py` to generate object detector on the video. Make sure to set `json_path` equal to `output_path` from the last step. 
