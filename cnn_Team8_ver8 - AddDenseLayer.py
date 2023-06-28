import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

'''
Functions to be used in the main program:

Check of dataset and count of image files in each class
Resize image with aspect ration preservation
Read image data
Perform one-hot encoding on a list of labels
Prepare data
Create CNN model
Train model using x_train and y_train (onehot encoded)
Create loss and accuracy plots
Test model
'''

'''
Check of dataset and count of image files in each class
'''
def check_data(data_dir):
    # Listing out the different classes
    filenames = os.listdir(data_dir)

    # List out potentially mislabelled filename
    mislabelled = []
    for filename in filenames:
        if filename[0] == '.':
            continue
        elif '_' not in filename:
            mislabelled.append(filename)
        else:
            continue
    if not mislabelled:
        print('Filenames in {} folder without underscore: none'.format(data_dir))
    else:
        print('Mislabelled filenames in {} folder: \n'.format(data_dir), mislabelled)

    # List out class names in 'fruits' array
    fruits = []

    for filename in filenames:
        if filename[0] == '.':
  
            continue
        elif filename.split('_')[0] not in fruits:
            fruits.append(filename.split('_')[0])
        else:
            continue

    print('Classes identified in {} folder: '.format(data_dir), fruits)

    # Count number of images in each class

    fruit_counts = {}

    for fruit in fruits:
        count = sum(1 for filename in filenames if fruit in filename.split('_')[0])
        fruit_counts[fruit] = count

    print('Number of files in each class in {}: '.format(data_dir))
    for fruit, count in fruit_counts.items():
        print(f"{fruit}: {count} files")

    print("\n")

'''
Resize image with aspect ratio preservation
'''

# function accepts a path to image and a target size tuple (width, height)
def resize_with_aspect_ratio(image, target_size):
    # Calculate the aspect ratio of the original image
    width, height = image.size
    aspect_ratio = width / float(height)

    # Determine the resizing dimension based on the aspect ratio
    target_width, target_height = target_size
    if (target_width / float(target_height)) > aspect_ratio:
        new_width = int(target_height * aspect_ratio)
        new_height = target_height
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)

    # Resize the image using the calculated dimensions
    img_resized = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Create a new RGB image with the target size
    img_rgb = Image.new("RGB", (target_width, target_height))
    # Paste the resized image onto the new RGB image
    img_rgb.paste(img_resized)

    # Return the resized image
    return img_rgb

'''
Read image data - change dimensions from 28x28 to what is required
'''
def read_img_data(data_dir):
    x = None #initialise x as None

    for file in os.listdir(data_dir):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory and resize
        img = Image.open("{}/{}".format(data_dir, file))
        img_resize = resize_with_aspect_ratio(img, (128,128))

        if x is None: #initialise x with first image
            x = np.array(img_resize)
        else: 
            #x = np.concatenate((x, img_resize))
            img_resize = np.array(img_resize)
            x = np.concatenate((x, img_resize))  # Concatenate subsequent images
            #np.newaxis is used to add a new dimension representing the batch axis when concatenating

    # Convert x_train to float32 datatype and normalize the pixel values to between 0 and 1
    x = x.astype('float32') / 255.0  

    # image is 28x28 and in color (3 channels)
    # -1 to let numpy computes the number of rows 
    return np.reshape(x, (-1,128,128,3)) 

'''
Perform one-hot encoding on a list of labels 
'''
def to_onehot(fruit): 
    # define class names and their index positions
    class_names = ["apple", "banana", "orange", "mixed"]
    # find index of the input fruit name within the class_names list
    try:
        # find index of the input fruit name within the class_names list
        class_index = class_names.index(fruit)
    except ValueError:
        # handle hidden files .DS
        return None
    # define onehot as a 1D array of 0s
    onehot = [0] * 4
    # create onehot encoding based on the fruit name
    onehot[class_index] = 1

    # return a numpy arrray
    return np.array(onehot)

'''
Prepare data
'''
def prep_data(data_dir):

    # obtain x_train as a numpy array, before obtaining y_train via onehot-encoding
    data = read_img_data(data_dir) #need to change this to read images one by one
    try:
        x = np.concatenate((x, data))
    except:
        x = data          

    # Construct the one-hot encodings for each class in data
    filenames = os.listdir(data_dir)
    y = []

    # iterates through each file, extracts the class name to perform onehot encoding
    for filename in filenames:
        if filename[0] == '.':  # skip hidden files
            continue
        class_name = filename.split("_")[0]
        y_onehot = to_onehot(class_name)
        y.append(y_onehot)

    y = np.array(y)
    return x, y

'''
Create CNN model
'''
def create_model():
    # create an empty neural network
    model = tf.keras.Sequential()

    # to adjust input shape and filters - test with filters=64, filters=128 and filters=256
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),  
                                     activation='relu', input_shape=(128,128,3))) 
    # add Avg Pooling Layer 
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))

    # add a flatten layer 
    model.add(tf.keras.layers.Flatten()) 
    # add dense layers 
    model.add(tf.keras.layers.Dense(units=128, activation='relu')) 
    # add new dense layer with 64 neurons
    model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    model.add(tf.keras.layers.Dense(units=4, activation='softmax')) 
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam',  
                  metrics=['accuracy']) 
                  
    return model

'''
Train model using x_train and y_train (onehot encoded)
'''

def train_model(model, x_train, y_train, x_val, y_val):

    # add weights
    sample_weights = compute_sample_weight('balanced', y_train)

    # creating an instance of ImageDataGenerator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images by 30 degrees - our team changed this parameter
        width_shift_range=0.1,  # randomly shift images horizontally by 0.1
        height_shift_range=0.1,  # randomly shift images vertically by 0.1
        zoom_range=0.1,  # randomly zoom images by 0.1
        horizontal_flip=True  # randomly flip images horizontally
    )

    # fit the generator on the training data
    datagen.fit(x_train)

    # train the model using augmented data
    return model.fit(datagen.flow(x_train, y_train, batch_size=32, sample_weight=sample_weights), 
                     epochs=30, validation_data=(x_val, y_val))

'''
Create loss and accuracy plots
'''
def plot(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

    ax[0].plot(hist.history['loss']) # training loss curve
    ax[0].plot(hist.history['val_loss']) # validation loss curve
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend(['Train', 'Validation'])

    ax[1].plot(hist.history['accuracy']) # training accuracy curve
    ax[1].plot(hist.history['val_accuracy']) # validation accuracy curve
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend(['Train', 'Validation'])

    plt.show()

'''
Test model
'''
def test_model(model, x_test, y_test):
    return model.evaluate(x=x_test, y=y_test)

'''
Main Program
'''
# run a check on train folder and test folder data
check_data('train')
check_data('test')

# create our CNN model
model = create_model()

# fetch training data and onehot-encoded labels
x_train, y_train = prep_data('train')
print(x_train.shape)
print(y_train.shape)

# split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size = 0.1, stratify=y_train)

# train model using training data
history = train_model(model, x_train, y_train, x_val, y_val)

# create loss and accuracy plots
plot(history)

# test model using test data and print result
x_test, y_test = prep_data('test')
result = test_model(model, x_test, y_test)
print(x_test.shape)
print(y_test.shape)
print('Loss and Accuracy: ', result)

