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
        help="Starts the Credit Card Fraud Detection Application"
    )
    divider()

# Actions on project
if start_project:
    with st.sidebar:
        time = st.slider("Time in seconds", min_value=0, max_value=172800, value=86400, step=1)
        v1 = st.slider("V1", min_value=0, max_value=100, value=50, step=1)
        v2 = st.slider("V2", min_value=0, max_value=100, value=50, step=1)
        v3 = st.slider("V3", min_value=0, max_value=100, value=50, step=1)
        v4 = st.slider("V4", min_value=0, max_value=100, value=50, step=1)
        v5 = st.slider("V5", min_value=0, max_value=100, value=50, step=1)
        v6 = st.slider("V6", min_value=0, max_value=100, value=50, step=1)
        v7 = st.slider("V7", min_value=0, max_value=100, value=50, step=1)
        v8 = st.slider("V8", min_value=0, max_value=100, value=50, step=1)
        v9 = st.slider("V9", min_value=0, max_value=100, value=50, step=1)
        v10 = st.slider("V10", min_value=0, max_value=100, value=50, step=1)
        v11 = st.slider("V11", min_value=0, max_value=100, value=50, step=1)
        v12 = st.slider("V12", min_value=0, max_value=100, value=50, step=1)
        v13 = st.slider("V13", min_value=0, max_value=100, value=50, step=1)
        v14 = st.slider("V14", min_value=0, max_value=100, value=50, step=1)
        v15 = st.slider("V15", min_value=0, max_value=100, value=50, step=1)
        v16 = st.slider("V16", min_value=0, max_value=100, value=50, step=1)
        v17 = st.slider("V17", min_value=0, max_value=100, value=50, step=1)
        v18 = st.slider("V18", min_value=0, max_value=100, value=50, step=1)
        v19 = st.slider("V19", min_value=0, max_value=100, value=50, step=1)
        v20 = st.slider("V20", min_value=0, max_value=100, value=50, step=1)
        v21 = st.slider("V21", min_value=0, max_value=100, value=50, step=1)
        v22 = st.slider("V22", min_value=0, max_value=100, value=50, step=1)
        v23 = st.slider("V23", min_value=0, max_value=100, value=50, step=1)
        v24 = st.slider("V24", min_value=0, max_value=100, value=50, step=1)
        v25 = st.slider("V25", min_value=0, max_value=100, value=50, step=1)
        v26 = st.slider("V26", min_value=0, max_value=100, value=50, step=1)
        v27 = st.slider("V27", min_value=0, max_value=100, value=50, step=1)
        v28 = st.slider("V28", min_value=0, max_value=100, value=50, step=1)
        amount = st.slider("Amount (Currency)", min_value=0, max_value=30000, value=15000, step=1)

    st.write(observations_on_data1)
    display_relationships()
    st.write(observations_on_data2)

    check_fraudulent = st.button("Make Prediction on Activity")
    if check_fraudulent:
        result = model.predict([[
            time, v1/100, v2/100, v3/100, v4/100, v5/100, v6/100, v7/100, v8/100, v9/100,
            v10/100, v11/100, v12/100, v13/100, v14/100, v15/100, v16/100, v17/100, v18/100,
            v19/100, v20/100, v21/100, v22/100, v23/100, v24/100, v25/100, v26/100, v27/100,
            v28/100, amount
        ]])

        st.write(result)

else:
    # Introducing the application
    st.write(index_introduction)

    st.markdown("###### Starting the Application")
    st.write("To start making use of this application, you should check the box on starting the application")
    load_image('images/start_application.png')

    st.markdown("###### Toggling the features")
    st.write("Since all the features on the data are continuous, you could toggle the various features on the data from the side panel")
    load_image('images/side_panel.png')

    st.markdown("###### Checking the predictions")
    st.write("Finally making predictions is now possible with the trained model with an assured **99.94%** accuracy")
    load_image('images/prediction_menu.png')
