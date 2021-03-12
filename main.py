import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import os
import random
import matplotlib.pyplot as plt

datadir = Path("/home/kalfasyan/data/images/fruit/")

# Helper functions
def load_image_array(image_path):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is not None :
            return image
        else :
            print(f"Could not read {image_path}")
            return np.array([])
    except Exception as e:
        print(f"Error loading image : {e}")
        return None

def get_label_from_image_path(image_path):
    return image_path.parts[-2]

# Defining a list of all training data paths and their labels
train_images = list(Path(os.path.join(datadir, 'Training')).rglob('*.jpg'))
train_labels = list(map(get_label_from_image_path, train_images)) 

# Grab 25 items randomly and plot them
k = 25 
random_idx = np.random.randint(0 , len(train_images), k)

plt.figure(figsize=(16,8))
for i, rdm in enumerate(list(random_idx)):
    # loading the image
    img = load_image_array(str(train_images[rdm]))
    # plotting it in a grid 5x5
    plt.subplot(5,5,i+1)
    plt.imshow(img);
    plt.title(train_labels[rdm])
    plt.axis('off')
plt.show()


# 