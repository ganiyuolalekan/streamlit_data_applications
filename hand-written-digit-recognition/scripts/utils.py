"""
Contains all the utilities functions
used for the app creation.
"""

from io import StringIO
import os
import joblib

import cv2
import streamlit as st

from PIL import Image


# Major Functions
def app_meta():
    """Adds app meta data to web applications"""

    # Set website details
    st.set_page_config(
        page_title="Hand Written Digit Recognition",
        page_icon="images/cc.jpeg",
        layout='centered'
    )


def create_absolute_path(dir_path):
    """Creates an absolute path given the provided directory"""

    return os.path.join(os.getcwd(), dir_path)


def divider():
    """Sub-routine to create a divider for webpage contents"""

    st.markdown("""---""")


def load_image(img_path):
    """Loads an image unto the site given it's path"""

    st.image(Image.open(create_absolute_path(img_path)))


# Model Handler
class ModelHandler:
    """
    Model handling functionality for the
    """

    def __init__(self, serialized_model=create_absolute_path('model/hwdr_model.z')):
        self.model = joblib.load(open(serialized_model, 'rb'))
        self.num_in_words = {
            0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
            5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'
        }

    @staticmethod
    def process_image(image):
        """Takes an image path, locate and preprocess it for the task"""

        return cv2.resize(
            cv2.imread(f"images/{image.name}", 0), (28, 28)
        ).reshape(1, 784)

    def predict(self, image):
        """Make a prediction given an image"""

        result = self.model.predict(self.process_image(image)).argmax(axis=1)[0]

        return result, self.num_in_words[result]


# Documentation
index_introduction1 = f"""# Handwritten Digit Classifier Application 

This application extends the notebooks we had on understanding and modeling the handwritten digit dataset. Here, we'll review the work done from the notebooks and then how this application can be used.

### About The Dataset

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. The problem we solved is the MNIST handwritten digit classification problem. which is a dataset of 70,000 (inclusive of the test data) hand-written numbers from the numbers 0-9. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

The MNIST handwritten digit is also provided in `tensorflow.keras.datasets.mnist`, which is what was used for this project as it has been curated for use in programming.

### The Data Modelling Process

The data model was created using TensorFlow's sequential model with two dense hidden layers and a linear activated dense output layer. The result was a model that performed with an accuracy of **98.93%** on the training set and **91.31%** on the validation set. A quick overview of the model summary."""


index_introduction2 = """After the model, the loss and accuracy were monitored to ensure the model wasn't overfitting or underfitting. The resulting model was saved and stored to be used in the application."""

observations_on_data1 = """Select an image of a digit from the images in the `../images/` directory."""
