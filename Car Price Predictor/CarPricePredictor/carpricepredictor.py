import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

st.title("🚗CAR PRICE PREDICTION🚗")
st.markdown("### THIS APPLICATION IS A CAR PRICE PREDICTOR")

@st.cache(persist=True)
def load_data(nrows):
    car_dataset = pd.read_csv('car data.csv')
    #encoding "Fuel_Type" Column
    car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
    #encoding "Seller_Type" Column
    car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
    #encoding "Transmission" Column
    car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
    return car_dataset

data=load_data(300)

if st.checkbox("Show Raw Data",False):
    st.subheader('Raw Data')
    st.write(data)

X = data.drop(['Car_Name', 'Selling_Price'],axis=1)
Y = data['Selling_Price']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,random_state=2)

def train(data):
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    return lin_reg_model

def trainL(data):
    lass_reg_model = Lasso()
    lass_reg_model.fit(X_train, Y_train)
    return lass_reg_model

def plots():
    fig, ax = plt.subplots(figsize=(width, height))
    plt.scatter(Y_train,training_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Prices vs Predicted Prices")
    st.pyplot(fig, clear_figure=None)

def plotts():
    fig, ax = plt.subplots(figsize=(width, height))
    plt.scatter(Y_test,test_data_prediction)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Prices vs Predicted Prices")
    st.pyplot(fig, clear_figure=None)


regressor = st.radio(
    "Which type of Regression??",
    ('Linear', 'Lasso'))


width = st.sidebar.slider("plot width", 1, 25, 10)
height = st.sidebar.slider("plot height", 1, 25, 5)
    

if regressor == 'Linear':
    trained = train(data)
    training_data_prediction = trained.predict(X_train)
    test_data_prediction = trained.predict(X_test)
    error_score = metrics.r2_score(Y_train, training_data_prediction)
    st.write("R squared Error(On training data): ", error_score)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    st.write("R squared Error(On test data): ", error_score)
    
    plott = st.radio(
    "Plot of:",
    ('Training Data', 'Test Data'))
    
    if plott == 'Training Data':
        plots()
    else:
        plotts()

    
else:
    trainedL = trainL(data)
    training_data_prediction = trainedL.predict(X_train)
    test_data_prediction = trainedL.predict(X_test)
    error_score = metrics.r2_score(Y_train, training_data_prediction)
    st.write("R squared Error(On training data): ", error_score)
    error_score = metrics.r2_score(Y_test, test_data_prediction)
    st.write("R squared Error(On test data): ", error_score)
    
    plott = st.radio(
    "Plot of:",
    ('Training Data', 'Test Data'))
    
    if plott == 'Training Data':
        plots()
    else:
        plotts()

    
