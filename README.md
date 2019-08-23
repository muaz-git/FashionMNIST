This repository (is a work in progress) focuses on solving
[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) using different architectures. Further details wil be
updated soon.


### Training and evaluation

Please run `python main.py -h` to understand the purpose of each argument.

#### MCDropout as a Bayesian Approximation 
Following [this paper](https://arxiv.org/pdf/1506.02142.pdf), dropouts remain on during evaluation time to have output
distribution instead of just a point-estimate. Add `--bayes=<number>` flag and `--percentile=<p>` to `main.py` to run. Where `<number>` is the number for
Monte Carlo Estimation and `<p>` is the percentile of maximum entropy to threshold the sample as uncertain. 


##### Object deteciton in video

To detect objects in a video, first generate some proposals using [imageai](https://github.com/OlafenwaMoses/ImageAI).
Follow [this tutorial](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606) to setup
imageai.

Note that choice of imageai is arbitrary and any other object detector library can be used. 
###### Proposal generator for the video
To get proposals run `python proposal_generator.py`. Make sure to set correct `model_path` and `video_path`. A dictionary
of all the bounding box (BB) proposals will be saved at the location `output_path`.
###### Testing on video
Once BB proposals have been generated. Run `python test_video.py` to generate object detector on the video. Make sure to
set arguments all the arguments. **Note** that video requires a model to be trained beforehand and weights should be provided
as an input argument to the script. 
