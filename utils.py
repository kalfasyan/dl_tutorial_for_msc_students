import cv2
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array

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

def check_str(string):
    m = re.search(r'\d+$', string)
    # if the string ends in digits m will be a Match object, or None otherwise.
    if m is not None:
        return m.string[:-2]
    else:
        return string

def train_generator(X_train, y_train, batch_size, nb_classes=10, img_dim=150):

    while True:
        for start in range(0, len(X_train), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_train))
            train_batch = X_train[start:end]
            labels_batch = y_train[start:end]
            
            for i in range(len(train_batch)): 
                data = cv2.imread(train_batch[i])
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = resize_image(data, size=(img_dim, img_dim))
                data = img_to_array(data) / 255.0

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32) 
            y_batch = np.array(y_batch, np.float32)
            y_batch = to_categorical(y_batch, nb_classes)

            yield x_batch, y_batch

def valid_generator(X_val, y_val, batch_size, nb_classes=9, img_dim=150):

    while True:
        for start in range(0, len(X_val), batch_size):
            x_batch = []
            y_batch = []

            end = min(start + batch_size, len(X_val))
            train_batch = X_val[start:end]
            labels_batch = y_val[start:end]
            
            for i in range(len(train_batch)): 
                data = cv2.imread(train_batch[i])
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                data = resize_image(data, size=(img_dim, img_dim))
                data = img_to_array(data) / 255.0

                x_batch.append(data)
                y_batch.append(labels_batch[i])

            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            y_batch = to_categorical(y_batch, nb_classes)

            yield x_batch, y_batch

def resize_image(img, size=(300,300)):
    # source: https://stackoverflow.com/a/49208362/3410400

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def get_labelencoder_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res