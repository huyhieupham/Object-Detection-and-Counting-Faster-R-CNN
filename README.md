# Object Detection and Counting Faster R-CNN


**Introduction**

[Faster R-CNN](https://arxiv.org/abs/1506.01497) is an state-of-the-art object detection algorithm proposed by Shaoqing Ren et al. in 2015. Faster R-CNN builds on previous work ([Fast R-CNN](https://arxiv.org/abs/1504.08083)) to efficiently detect and classify object proposals using deep convolutional neural networks (D-CNNs). Compared to Fast R-CNN, Faster R-CNN employs a region proposal network and does not require an external method for candidate region proposals. Therefore, the time cost of generating region proposals in Faster R-CNN is much smaller than selective search in Fast R-CNN. This feature is very important for building real-time computer vision applications.

A vision-based tool for transport system flow analysis using Faster R-CNN. This system is able to detect different types of objects, e.g., cars, buses, pedestrians, and classify as well as count them in transport videos with a real-time speech.
