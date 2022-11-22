# The Hand Written Digit Recognition Application

In this project we analyzed the hand written digit recognition dataset in the [notebooks](notebooks). During the analysis we (1) looked into the dataset to [understanding it](notebooks/understanding_the_dataset.ipynb) and we then (2) [modeled the dataset](notebooks/modelling_the_data.ipynb). The result from the modelling process gave us a model that performed with an accuracy of **98.93%** on the training set and **91.31%** on the validation set.

### About the Dataset

The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset. The problem we solved is the MNIST handwritten digit classification problem. which is a dataset of 70,000 (inclusive of the test data) hand-written numbers from the numbers 0-9. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.

### Running the Application

To run the application, ensure first that the requirements for the project are all installed, and that you have pip and python running on your system, then run

```shell
pip install -r requirement.txt
```

then ensure you navigate the `hand-written-digit-recognition/` folder and run the command

```shell
streamlit run scripts/apps.py
```

With this the streamlit application interface will automatically launch on your system with guide on how to navigate the application.
