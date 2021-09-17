import cv2
import datetime as dt
import h5py
import os
from glob import glob



def proc_images(main_path,path_list,output_path):                   # The labels need to be assigned according to the files in each of the directories here 
    """
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class label
    """
    start = dt.datetime.now()
    
    with h5py.File(output_path + 'data.h5', 'w') as hf:
        count = 0 
        for j,p in enumerate(path_list):
            images = glob(os.path.join(p, "*.jpg"))      # jpg images only here 
            for i,img in enumerate(images):
                image = cv2.imread(img)
                
                HEIGHT, WIDTH, CHANNELS = image.shape
                
                Xset = hf.create_dataset(
                    name='X'+str(count+i),
                    data=image,
                    shape=(HEIGHT, WIDTH, CHANNELS),
                    maxshape=(HEIGHT, WIDTH, CHANNELS),
                    compression="gzip",
                    compression_opts=9)
                
                yset = hf.create_dataset(
                name='y'+str(count+i),
                data=j,
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)

            count+=(i+1)                   # Newer indices to be generated here 
#             print(count)

                
                
    end=dt.datetime.now()
    print("\n Time required is \n")
    print("\r", i, ": ", (end-start).seconds, "seconds", end="")


def gen_hdf5_file(main_path="/home/bhushan/Desktop/bhushan_env/bhushan/Attention in CV/attn_cv_2/flowers_resized/",output_path="/home/bhushan/Desktop/bhushan_env/bhushan/Attention in CV/attn_cv_2/output_hdf5_flowers/"):
    path_list = []
    for f in os.listdir(main_path):
        path = main_path + f + "/"
        path_list.append(path)
    
    
    proc_images(main_path,path_list,output_path)

gen_hdf5_file()
