# WaterCam

![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

A camera for segmenting water images in the field.

<img src="./assets/FieldTest.jpg" width=100% height=100%/>


The device is a Raspberry Pi 4, Hyperpixel display, Camera Module 3, and an on-board ML model that was made with the [RIWA dataset](https://www.kaggle.com/dsv/4289421) and [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym).

Each picture takes ~3 seconds to segment and display on screen. Future work is focused on decreasing this latency.

<img src="./assets/pic.jpg" width=50% height=50%/>

<img src="./assets/overlay.jpg" width=50% height=50%/>
