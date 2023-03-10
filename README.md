# WaterCam

![Raspberry Pi](https://img.shields.io/badge/-RaspberryPi-C51A4A?style=for-the-badge&logo=Raspberry-Pi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

A camera for segmenting water images in the field.

<img src="./assets/FieldTest.jpg" width=60% height=60%/>


The device is a Raspberry Pi 4, Hyperpixel display, Camera Module 3, and an on-board ML model that was made with the [RIWA dataset](https://www.kaggle.com/dsv/4289421) and [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym).

Using a Google Coral edgeTPU, a 256 x 256 model can run at 3 frames/second.
Without an accelerator, pictures takes ~3 seconds to segment.

<img src="./assets/pic.jpg" width=50% height=50%/>

<img src="./assets/overlay.jpg" width=50% height=50%/>
