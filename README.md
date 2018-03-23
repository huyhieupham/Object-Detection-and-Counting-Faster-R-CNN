# Object Detection and Counting Faster R-CNN


## Introduction

[Faster R-CNN](https://arxiv.org/abs/1506.01497) is an state-of-the-art object detection algorithm proposed by Shaoqing Ren et al. in 2015. Faster R-CNN builds on previous work ([Fast R-CNN](https://arxiv.org/abs/1504.08083)) to efficiently detect and classify object proposals using deep convolutional neural networks (D-CNNs). Compared to Fast R-CNN, Faster R-CNN employs a region proposal network and does not require an external method for candidate region proposals. Therefore, the time cost of generating region proposals in Faster R-CNN is much smaller than selective search in Fast R-CNN. This feature is very important for building real-time computer vision applications.


Basically, a Faster R-CNN uses a *Region Proposal Network (RPN)* to generate high-quality region proposals, which are used for detection network as ([Fast R-CNN](https://arxiv.org/abs/1504.08083)). In other words, [Faster R-CNN](https://arxiv.org/abs/1506.01497) is the combination between RPN and ([Fast R-CNN](https://arxiv.org/abs/1504.08083)). 

 <p align="center"> 
<img src="https://github.com/huyhieupham/Object-Detection-and-Counting-Faster-R-CNN/blob/master/figure/Faster-RCNN.png">
</p>

The unified network of Faster R-CNN proposed by [Faster R-CNN](https://arxiv.org/abs/1506.01497). The first component is a deep fully convolutional network that proposes region proposals. The second component is the [Fast R-CNN](https://arxiv.org/abs/1504.08083) detector.


In this project, we use [Faster R-CNN](https://arxiv.org/abs/1506.01497) to build a vision-based tool for transport system flow analysis. This system is able to detect different types of objects, e.g., cars, buses, pedestrians, and classify as well as count them in transport videos with a real-time speech.

## Requirements

This project replies on [Keras 2.0.7](https://faroit.github.io/keras-docs/2.0.7/), [TensorFlow 1.0.3](https://www.tensorflow.org/install/), [OpenCV-Python 3.3.0](https://pypi.python.org/pypi/opencv-contrib-python/3.3.0.10), [scikit-learn 0.19.0](https://pypi.python.org/pypi/scikit-learn/0.19.0), [h5py 2.7.0](https://pypi.python.org/pypi/h5py/2.7.0), and [numpy 1.13.1](https://pypi.python.org/pypi/numpy/1.13.1). More details about the instalation these libraies on Windows can be found [here](https://github.com/huyhieupham/Installing-Keras-Theao-Tensorflow-with-GPU-on-Windows-10).

## Get started 

Training deep neural networks is time-consumming. To use this tool directy on unseen videos, you can dowload pre-trained model [here]( https://drive.google.com/open?id=1xNjHc2bLwRc_HkqwtIwIKApP50r0-vOw). Putting it into the working directory, opennining command prompt  and performing the inference stage by command:

```cd path-to-the-working-directory```

```inference.py --input_file path-to-input-folder\input-video.mp4 --output_file path-to-output-folder\output-video.mp4 --frame_rate=30```


https://www.youtube.com/watch?v=H6Q7f-zGnNM
