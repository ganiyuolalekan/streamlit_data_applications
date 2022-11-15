"""
Contains all the utilities functions
used for the app creation.
"""

import os
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from PIL import Image
from matplotlib import pyplot as plt


# Initialization
target_column = 'Class'
data_path = "data/creditcard.csv"

credit_card_data = pd.read_csv(data_path)

credit_card_data['Class'] = credit_card_data['Class'].map({0: 'Not Fraudulent', 1: 'Fraudulent'})
credit_card_data['Hour'] = credit_card_data['Time'].apply(lambda x: np.floor(x / 3600))

tmp = credit_card_data.groupby(['Hour', 'Class'])['Amount'].aggregate(
    ['min', 'max', 'count', 'sum', 'mean', 'median', 'var']
).reset_index()


# Major Functions
def app_meta():
    """Adds app meta data to web applications"""

    # Set website details
    st.set_page_config(
        page_title="CreditCard Fraud Detection",
        page_icon="images/cc.jpeg",
        layout='centered'
    )


def create_absolute_path(dir_path):
    """Creates an absolute path given the provided directory"""

    return os.path.join(os.getcwd(), dir_path)


def display_relationships():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    features = [
        'min', 'max', 'count',
        'sum', 'mean', 'median'
    ]

    for ax, feature in zip(axes.ravel(), features):
        sns.lineplot(
            data=tmp, x='Hour', y=feature,
            hue=target_column, ax=ax
        )
        ax.set_title(feature)

    plt.title("Data Distribution Across the Features")
    st.pyplot(fig)


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

    def __init__(self,
                 serialized_model=create_absolute_path('model/creditcardmodel.z')
                 ):
        self.model = joblib.load(open(serialized_model, 'rb'))
        self.target_identifier = {
            0: "Not Fraudulent",
            1: "Fraudulent"
        }
        self.columns = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7',
            'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
            'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23',
            'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]

    def predict(self, values):
        assert type(values) == list, "Expected values should be a list type"

        data = pd.DataFrame(values, columns=self.columns)

        dis_data = data.T

        st.write(dis_data)

        return self.target_identifier[self.model.predict(data)[0]]


# Documentation
index_introduction = """# Credit Card Fraud Detection Application 

This application extends the notebooks we had on understanding and modeling the credit card dataset. Here, we'll review the work done from the notebooks and then the use of this application.

### About The Dataset

According to the source; **The credit card fraud detection dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.**

The dataset from the source is highly unbalanced, the positive class (1, frauds) accounts for 0.172% of all transactions. It contains only numerical input variables resulting from a PCA transformation.

The numeric PCA features in the dataset account for the sensitive data features; thus, dimensionality reduction strategies were applied to encode these features while preserving the feature's meaning. These features are labeled v1 to v28 in the dataset

> Check out the notebooks on [understanding the dataset](../notebooks/understanding_the_dataset.ipynb) and [modeling the dataset](../notebooks/modelling_the_data.ipynb) for more observations on the credit card fraud detection dataset.

But for quick reference, we'll be talking a bit about how the data was modeled.

### The Data Modelling Process

The features present in the dataset include;

1. `Time`: Recorded in seconds for a period of two (2) days. These features were imputed in seconds. Thus, the value ranges from 0 - 172,800.
2. `v1`-`v28`: Representing the encoded features containing sensitive data that couldn't be made available publicly. In the [notebook](../notebooks/modelling_the_data.ipynb#Adding-Implementation-Criteria) we scaled these values from 0-1 for easier implementation on the web application.
3. `Amount`: The transaction amount for each transaction. This feature wasn't affected in any way but ranges from the value of 0 to 30,000.

### Overview of the Application
There are three (3) areas to making use of this application"""

observations_on_data1 = """### Observations on Fraudulent and Non-Fraudulent Classes

During the data modeling steps, some of the observations noticed that distinguished both fraudulent and non-fraudulent classes can be reviewed in the graphs below.

we aggregated the data on a new feature column - `Hour` (which was derived from the `Time` feature column). Then this feature alongside the `Class` feature (target column) was aggregated to get the `Min`, `Max`, `Count`, `Sum`, `Mean`, `Median`, and `Variance`. These features were than used to derive this relationship between the data."""

observations_on_data2 = """
These observations can be used to experiment and make other predictions using the sidebar to amend the features."""
