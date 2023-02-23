import os
from glob import glob
import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from skimage.transform import resize


#set model path
filepath = 'weights/RIWA256_fullmodel_model'

##### 

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

def load_npz(example):
    with np.load(example.numpy()) as data:
        image = data['arr_0'].astype('uint8')
        image = standardize(image)
        label = data['arr_1'].astype('uint8')

    return image, label

@tf.autograph.experimental.do_not_convert
def read_seg_dataset_multiclass(example):
    """
    "read_seg_dataset_multiclass(example)"
    This function reads an example from a npz file into a single image and label
    INPUTS:
        * dataset example object (filename of npz)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    image, label = tf.py_function(func=load_npz, inp=[example], Tout=[tf.float32, tf.uint8])
    label = tf.expand_dims(label,-1)

    #label = tf.expand_dims(label,-1)

    return image, label
######
#make representative Dataset
#filenames = tf.io.gfile.glob('/npz4gym/RIWA_actual_256_'+'*noaug*.npz')
#if len(filenames)==0:
file_pattern = 'npz4gym/RIWA_actual_256_noaug*.npz'

filenames = tf.io.gfile.glob(file_pattern)
print(filenames)
list_ds = tf.data.Dataset.list_files(filenames, shuffle=False)
train_ds = list_ds.take(int(len(filenames)))
train_ds = train_ds.map(read_seg_dataset_multiclass, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(1, drop_remainder=True) # drop_remainder will be needed on TPU (and possible with distributed gpus)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE) 

print(train_ds.element_spec)


def representative_dataset():
    for data,_ in train_ds.take(1000):
        yield [tf.dtypes.cast(data, tf.float32)]

#####
#INT Quantization
#TF lite converter 
converter = tf.lite.TFLiteConverter.from_saved_model(filepath)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

#save tflite  model
tflite_models_dir = pathlib.Path("./tflite_model/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir/"RIWA265_feb2023_INT.tflite"
tflite_model_file.write_bytes(tflite_model)


#Load model into TFlite intepreter
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)


##### 
#Sanity check on image using Int 
interpreter.allocate_tensors()

#SET THE IMAGE size
pix_dim = 256
imsize = (pix_dim, pix_dim) 

#Prep the input
imgp = "images/img_0139.jpg"
img = tf.keras.preprocessing.image.load_img(imgp,target_size = imsize)
img = tf.keras.preprocessing.image.img_to_array(img)
Simg = standardize(img)
test_image = np.expand_dims(Simg,axis=0)
#test_image = tf.convert_to_tensor(test_image, dtype=tf.int8)


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

