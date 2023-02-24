from picamera2 import Picamera2, Preview
from libcamera import controls

from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

import time
import numpy as np

#SET UP TF MODEL

path_to_model = "/home/pi/Desktop/RIWA265_feb2023_INT_edgetpu.tflite"
interpreter = edgetpu.make_interpreter(path_to_model)

interpreter.allocate_tensors()

#Doodleverse standardization using adjusted standard deviation
def standardize(img):

    N = np.shape(img)[0] * np.shape(img)[1]
    s = np.maximum(np.std(img), 1.0/np.sqrt(N))
    m = np.mean(img)
    img = (img - m) / s
    del m, s, N
    #
    if np.ndim(img)==2:
        img = np.dstack((img,img,img))

    return img

#prediction fn
def TFlitePred(img):
    converted = np.array(img, dtype=np.float32)
    st_img = standardize(converted)
    img_exp = np.expand_dims(st_img, axis=0)

    common.set_input(interpreter, img_exp)
    interpreter.invoke()
    
    predictions = common.output_tensor(interpreter, 0)

    pred_sq = predictions.squeeze()
    pred = np.argmax(pred_sq,-1)
    

    return pred


# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start(show_preview=True)

while True:
    start=time.time()
    image =picam2.capture_image("main").convert('RGB')
    image = image.resize((256, 256))
    


    mask = TFlitePred(image)

    overlay = np.zeros((256,256,4),dtype=np.uint8)

    #broadcast
    overlay[mask == 1] = [255,255,0,64]
    overlay[mask == 0] = [0,0,0,64]

    picam2.set_overlay(overlay)
    done = time.time()
    print(done-start)
