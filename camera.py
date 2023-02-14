from tkinter import *
from picamera2 import Picamera2, Preview
from libcamera import controls
from time import sleep
from PIL import Image, ImageTk
from skimage.transform import resize


import matplotlib.pyplot as plt
import numpy as np
from tflite_runtime.interpreter import Interpreter

#SET UP TF MODEL

path_to_model = "/home/pi/Desktop/RIWA256_feb2023.tflite"
interpreter = Interpreter(path_to_model)
interpreter.allocate_tensors()
#print(interpreter.get_input_details())

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
    #get image in the correct shape,size, format
    #re_img = img.resize((512,512))
    
    converted = np.array(img, dtype=np.float32)
    st_img = standardize(converted)
    img_exp = np.expand_dims(st_img, axis=0)
    #print(img_exp.shape)
    
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, img_exp)
    
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_index)

    pred_sq = predictions.squeeze()
    pred = np.argmax(pred_sq,-1)
    print(pred)

    return pred


# Initialize the camera
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

# Define a function to capture a photo and display it
def capture():
    # Take a photo
    #picam2.start_preview(Preview.QTGL)
    picam2.start()
    sleep(2)
    picam2.capture_file("test.jpg")
    #picam2.stop_preview()
    picam2.stop()

    # Load the photo into a PIL Image object and resize it
    image = Image.open('/home/pi/Desktop/test.jpg')
    image = image.resize((256, 256))
    
    mask = TFlitePred(image)
    #print(mask)
    plt.imshow(image,cmap='gray')
    plt.imshow(mask, alpha=0.4)
    plt.axis("off")
    plt.margins(x=0, y=0)
    #plt.show()
    plt.savefig("overlay.jpg", bbox_inches="tight")

    # Convert the PIL Image to a tkinter PhotoImage and display it on the screen
    overlay = Image.open('/home/pi/Desktop/overlay.jpg')
    photo = ImageTk.PhotoImage(overlay)
    label.config(image=photo)
    label.image = photo

# Set up the tkinter window
root = Tk()
root.geometry('500x500')
root.title('Raspberry Pi Camera GUI')

# Add a label to display the captured photo
label = Label(root)
label.pack()

# Add a button to capture a photo
button = Button(root, text='Capture', command=capture)
button.pack()

# Start the tkinter event loop
root.mainloop()
