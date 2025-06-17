import os
import cv2
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def load_video(in_dir, in_file, max_frames=0):
    # file_name = in_file.split("/")[-1].replace(".avi", "")
    file_name = os.path.basename(in_file).replace(".avi", "").replace(".mp4", "")
    save_dir = os.path.join(in_dir, file_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vidcap = cv2.VideoCapture(in_file)
    success,image = vidcap.read()
    idx = 0

    try:
        while success:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dim = (224, 224)
            frame = cv2.resize(img, dim)
            cv2.imwrite( os.path.join(save_dir, "frame"+"_"+str(idx)+".jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            success,image = vidcap.read()
            idx += 1
    finally:
        vidcap.release()
    return

# ====== CONFIGURATION ======
dataset_path = "datasets/my_data"
save_path = "datasets/my_data_frames"

splits = ['train', 'val']
categories = ['Fight', 'NonFight']

for split in splits:
    for category in categories:
        input_dir = os.path.join(dataset_path, split, category)
        output_base = os.path.join(save_path, split, category)
        if not os.path.exists(output_base):
            os.makedirs(output_base)

        print(f"Processing {split}/{category}...")
        files = [f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f.endswith(('.avi', '.mp4'))]
        for f in tqdm(files):
            in_file = os.path.join(input_dir, f)
            load_video(output_base, in_file)
'''
dataset_path = "2 - хИТхИЖш┐ЗчЪДцХ░цНощЫЖ"
dict_names = {}

for root, dir, files in os.walk(dataset_path+"/train", topdown=False):
  length = len(root.split("/"))
  if(length>3):
    key = ((root.split("/")[3]))
    dict_names[key] = []
    for f in listdir(root):
      if isfile(join(root, f)):
        dict_names[key].append(join(root, f))

# Data path rewrite
base_dir = "datasets/RWF-2000_frames"
save_dir_train = os.path.join(base_dir, "train")
if not os.path.exists(save_dir_train):
        os.makedirs(save_dir_train)

for key in dict_names:
  savedir = os.path.join(save_dir_train, key)
  if not os.path.exists(savedir):
        os.makedirs(savedir)
  for elem in tqdm(dict_names[key]):
    load_video(savedir , elem )

dict_names_validation = {}
for root, dir, files in os.walk(dataset_path+"/val", topdown=False):
  length = len(root.split("/"))
  if(length>3):
    key = ((root.split("/")[3]))
    dict_names_validation[key] = []
    for f in listdir(root):
      if isfile(join(root, f)):
        dict_names_validation[key].append(join(root, f))

base_dir = "datasets/RWF-2000_frames"
save_dir_val = os.path.join(base_dir, "val")
if not os.path.exists(save_dir_val):
        os.makedirs(save_dir_val)

for key in dict_names_validation:
  savedir = os.path.join(save_dir_val, key)
  if not os.path.exists(savedir):
        os.makedirs(savedir)
  for elem in tqdm(dict_names_validation[key]):
    load_video(savedir , elem )
'''