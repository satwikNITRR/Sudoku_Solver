import cv2
import os
import numpy as np

project_path=r"C:\Users\Asus\Desktop\NeuroDoku"
sudoku_images=os.path.join(project_path,"sudoku_images")
data_list_path=os.path.join(sudoku_images,r"dataset")
data_list=os.listdir(data_list_path)
data_X=[]
data_Y=[]

count=1
for image_name in data_list:
    if image_name.lower().endswith(".jpg"):
        print(count)
        image_path=os.path.join(data_list_path,image_name)
        pic=cv2.imread(image_path)
        pic=cv2.resize(pic,(32,32))
        data_X.append(pic)
        count+=1


data_X=np.array(data_X)
data_Y=np.array(data_Y)

def Prep(img):
    if len(img.shape) > 2 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to grayscale
    img = img.astype(np.uint8) # convert to 8-bit unsigned integer
    img = cv2.equalizeHist(img) # histogram equalization
    img = img/255 # normalizing
    return img