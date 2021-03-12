import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import get_label_from_image_path, load_image_array, check_str

datadir = Path("/home/kalfasyan/data/images/fruit/")
# Inside my "fruit" folder I have "Test" and "Training" from the downloaded archive

# Defining a list of all training data paths and their labels
train_images = list(Path(os.path.join(datadir, 'Training')).rglob('*.jpg'))
train_labels = list(map(get_label_from_image_path, train_images)) 
train_labels = pd.Series(train_labels).apply(lambda x: check_str(x)) # ignore this; I just reduced the number of classes by renaming e.g. Pear 2 => Pear

test_images = list(Path(os.path.join(datadir, 'Training')).rglob('*.jpg'))
test_labels = list(map(get_label_from_image_path, train_images)) 
test_labels = pd.Series(test_labels).apply(lambda x: check_str(x)) # ignore this; I just reduced the number of classes by renaming e.g. Pear 2 => Pear

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


# Splitting the data into training / validation
# since we already have "Test" data in a separate folder
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)