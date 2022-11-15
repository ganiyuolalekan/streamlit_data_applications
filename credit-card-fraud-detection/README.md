# The Credit Card Fraud Detection Application

In this project we analyzed the credit card fraud detection dataset in the [notebooks](notebooks). During the analysis we (1) looked into the dataset to [understanding it](notebooks/understanding_the_dataset.ipynb) and we then (2) [modeled the dataset](notebooks/modelling_the_data.ipynb). The result from the modelling process gave us a model that performed with an accuracy of 99.94%.

### About the Dataset

The dataset which was sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), is a fraud detection dataset that contains transactions made by credit cards in September 2013 by European cardholders. 

To test-run the dataset on the application, you should get the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and add it to a folder called data directly under [credit-card-fraud-detection](credit-card-fraud-detection). The data folder was safely excluded when uploading because of it's size.

This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is **highly unbalanced**, the positive class (frauds) account for **0.172%** of all transactions.

The numeric PCA features in the dataset account for the sensitive data features; thus, dimensionality reduction strategies were applied to encode these features while preserving the feature's meaning. These features are labeled v1 to v28 in the dataset

Feature `Time` contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature `Amount` is the **transaction Amount**. Feature **'Class'** is the response variable and it takes **value 1 in case of fraud and 0 otherwise**.

### Running the Application

To run the application, ensure first that the requirements for the project are all installed, and that you have pip and python running on your system, then run

```shell
pip install -r requirement.txt
```

then ensure you navigate the `credit-card_fraud-detection/` folder and run the command

```shell
streamlit run scripts/apps.py
```

With this the streamlit application interface will automatically launch on your system with guide on how to navigate the application.
