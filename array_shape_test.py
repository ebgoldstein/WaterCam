import os
import numpy as np

# Specify the path to the folder containing npz files
folder_path = './data/npz4gym'

# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.npz'):  # Check if file is npz file
        # Load data from npz file
        data = np.load(os.path.join(folder_path, file_name))
        
        # Check if 'arr_0' variable has shape (512, 512, 3)
        if 'arr_0' in data.files and data['arr_0'].shape == (512, 512, 3):
            a = 1+1
            #print(f"'arr_0' in file '{file_name}' has the correct shape.")
        else:
            print(f"WARNING: 'arr_0' in file '{file_name}' does not have the correct shape. It is being deleted.")
            os.remove(os.path.join(folder_path, file_name))