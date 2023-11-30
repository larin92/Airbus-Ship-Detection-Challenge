import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.measure import label, regionprops
import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array, load_img
from train import WORK_DIR, IMAGE_SIZE_TRAINING

# Load the model
model = load_model(os.path.join(WORK_DIR, 'model.h5'))


def _preprocess_image(image_path):
    # Preprocess the image for model prediction
    # This function should take into account the preprocessing steps used during model training
    image = load_img(image_path, target_size=IMAGE_SIZE_TRAINING, color_mode='rgb')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0  # Rescale pixel values

    return image


def _postprocess_mask(predicted_mask):
    # Post-process the mask from model prediction
    # This might include thresholding and converting to a suitable image format
    thresholded = (predicted_mask > 0.5).astype(np.uint8)  # Apply a threshold to convert probabilities to binary mask
    
    return thresholded


def predict(image_path):
    # Use the model to predict the image's mask
    x = _preprocess_image(image_path)
    predicted_mask = model.predict(x)
    
    return _postprocess_mask(predicted_mask)


def evaluate(test_data):
    # Evaluate the model on the test data
    # This function should return the loss and accuracy metrics
    loss, accuracy = model.evaluate(test_data)
    
    return loss, accuracy
