import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from skimage.transform import resize


#set model path
filepath = './data/weights/RIWA_fullmodel_model'

#TF lite converter 
converter = tf.lite.TFLiteConverter.from_saved_model(filepath)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

#save tflite  model
tflite_models_dir = pathlib.Path("./tflite_model/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"RIWA_feb2023.tflite"
tflite_model_file.write_bytes(tflite_model)

#Sanity check on image 
#Load model into TFlite intepreter
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
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

#SET THE IMAGE size
pix_dim = 512
imsize = (pix_dim, pix_dim) 

#Prep the input
imgp = "./data/images/img_0139.jpg"
img = tf.keras.preprocessing.image.load_img(imgp,target_size = imsize)
img = tf.keras.preprocessing.image.img_to_array(img)
Simg = standardize(img)
test_image = np.expand_dims(Simg,axis=0)

#get & set the tflite model details, make the prediction
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
#print(interpreter.get_output_details())
interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

#plot the results - image and then predcition
img1 = plt.imread(imgp)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img1)
axs[0].grid(False)


pred_sq = predictions.squeeze()
label = np.argmax(pred_sq,-1)
label_resized = resize(label, img1.shape[:2], preserve_range=True)
axs[1].imshow(label_resized)
axs[1].grid(False)

# Show the plot
plt.show()

