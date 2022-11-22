"""
The Application Interface for the program

This interface introduces how to use the
"""

from utils import *

# Adds the application metadata
# and prepares application model
app_meta()
model = ModelHandler()

# Initializing Side-Bar
with st.sidebar:
    st.write("")
    start_project = st.checkbox(
        label="Start Application",
        help="Starts the Hand Written Digit Recognition Application"
    )
    divider()

# Actions on project
if start_project:
    with st.sidebar:
        image = st.file_uploader("Upload an image of a single digit")

    st.write(observations_on_data1)
    divider()

    make_prediction = st.button(
        "Make Prediction on Image",
        help="Ensure image contains a single digit"
    )
    if make_prediction:
        num, word = model.predict(image)

        load_image(f"images/{image.name}")

        st.write(f"Model predicted digit as {num} - {word}")

else:
    # Introducing the application
    st.write(index_introduction1)
    load_image('images/model_summary.png')
    st.write(index_introduction2)
    load_image('images/loss_accuracy.png')

    st.markdown("###### Starting the Application")
    st.write("To start making use of this application, you should check the box on starting the application")
    load_image('images/start_application.png')

    st.markdown("###### Image Upload")
    st.write("")
    load_image('images/side_panel.png')

    st.markdown("###### Checking the predictions")
    st.write("Finally making predictions is now possible with the trained model with high accuracy")
    load_image('images/prediction_menu.png')
