# ML Prediction App 
# Use selectbox to select input data 


import numpy as np
import pandas as pd
import re
import pickle
from sklearn.ensemble import RandomForestClassifier 

import streamlit as st


st.set_page_config(
    layout = "wide", 
    initial_sidebar_state="auto",
    page_title=None,
    page_icon=None)
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option('display.max_colwidth', None)

""
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    width: 275px;
    margin-left: -5px;}

    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    margin-right: -50px;}
    </style>
    """,
    unsafe_allow_html=True)

#########################################################################
# Utility Function for the Sidebar
st.sidebar.header("Input Data")
dataSelect = st.sidebar.selectbox("", 
    ["<Click to Select>","Upload File","Manual Select"], 0)

if dataSelect=="<Click to Select>":
    upload_flag = 0
    #st.warning("The Data Needs to be Selected")

elif dataSelect=="Upload File":
    uploaded_file = st.sidebar.file_uploader("", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        upload_flag = 1

else:
    Gender = st.sidebar.selectbox('Gender',('Female','Male'))
    Mortgage = st.sidebar.selectbox('Mortgage',['N','Y'],1)
    MaritalStatus = st.sidebar.selectbox('MaritalStatus',['Married','Other','Single'],1)
    Loans = st.sidebar.selectbox('Loans', [0,1,2,3],3)
    Education = st.sidebar.selectbox('Education', [1,2,3], 1)
    Age = st.sidebar.slider('Age', 15, 70, 34)
    Income = st.sidebar.slider('Income', 15300.0, 78400.0, 50000.0)
    Experience = st.sidebar.slider('Experience', 1, 15, 11)
    data = {'Gender': Gender,
            'MaritalStatus': MaritalStatus,
            'Mortgage': Mortgage,
            'Loans': Loans,
            'Education': Education, 
            'Age': Age,
            'Income': Income,
            'Experience': Experience
            }
    input_df = pd.DataFrame(data, index=[0])
    upload_flag = 2

#########################################################################
# Main Page 
# st.markdown(
#     """
#     # Prediction App 
#     ### Customer Risk Profile
#     <font color='red'> Created by Nurur Rahman </font>
#     """, 
#     unsafe_allow_html=True)

mkdtext = """
        # Prediction App 
        ### Customer Risk Profile
        <font color='red'> Created by Nurur Rahman </font>
        """
st.markdown(mkdtext, unsafe_allow_html=True)

st.info(
    """
    App Properties: \n
    1. Predict risk types of customers using a pre-built model.
    2. A Random Forest model has been trained and serialized as a pickle object.
    3. The model is uploaded before it is applied for prediction.
    4. A test data file is read from the local drive.
    5. If no file is selected from the local drive, then the data is selected manually. 
    """
)

from utilityDataProperty import get_dataProperty

try:    
    input_shape = input_df.shape
    #st.write(f"The shape of input_df {input_shape}")

    # Combine uploaded test data with the original train data
    original_df = pd.read_csv('train_riskClass.csv')
    original_df = original_df.drop( columns=['Risk'] )
    df = pd.concat([input_df, original_df], axis=0)
    del original_df

    # Encoding of ordinal features
    df_cat = df[['Gender','MaritalStatus','Mortgage']]
    colcat = df_cat.columns.tolist()
    df_dum = pd.get_dummies( df_cat, prefix= colcat )
    df = pd.concat( [df, df_dum], axis=1)
    df = df.drop(columns=colcat)
    del df_cat, colcat, df_dum 

    # Select rows consistent with input_df
    df = df[ :input_shape[0] ] 


# Displays uploaded data or user input data
    if upload_flag==0:
        st.warning("The Data Needs to be Selected")

    elif upload_flag==1:
        get_dataProperty()
        st.write(" ")
        st.info("Uploaded Data")

        showButton = st.button("Click to View the Data")
        if showButton :
            st.write( df )
            hideCheckBox = st.checkbox("Click to Hide the Data")
            if hideCheckBox:
                showButton = False

    else:
        get_dataProperty()
        st.write(" ")
        st.info('User Input Data')
        showButton = st.button("Click to View the Data")
        if showButton :
            st.write( df )

            hideCheckBox = st.checkbox("Click to Hide the Data")
            if hideCheckBox:
                showButton = False

    #######################  Model Prediction 
    # Read the saved model
    load_clf = pickle.load( open('model_riskClass.pkl', 'rb') )

    # Apply model to make predictions
    pred_class = load_clf.predict(df)
    pred_proba = load_clf.predict_proba(df)

    st.info('Prediction Probability for the Input Features')
    st.write(pred_proba)

    st.info('Prediction Class for the Input Features')
    risk_class = np.array(['Good Risk','Bad Risk'])
    st.write(risk_class[pred_class])

except NameError:
   print("The Test Data Needs to be Selected")
