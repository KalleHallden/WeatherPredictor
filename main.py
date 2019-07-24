
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
import csv
import datetime
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression


dataset_url1 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/corrected-archive/data.csv'
dataset_url2 = 'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/2/station/71420/period/latest-months/data.csv'
data1 = pd.read_csv(dataset_url1, sep=';', skiprows=3607, names= [
    'Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'
])
data2 = pd.read_csv(dataset_url2, sep=';', skiprows=15, names= [
    'Fran Datum Tid (UTC)', 'till', 'day', 'temperature', 'Kvalitet', 'Tidsutsnitt:', 'Unnamed: 5'
])

def train_data():
    x = data1.drop('Kvalitet', axis = 1)
    x = x.drop('Unnamed: 5', axis = 1)
    x = x.drop('Fran Datum Tid (UTC)', axis = 1)
    x = x.drop('Tidsutsnitt:', axis = 1)
    y = x.temperature
    X = x.drop('temperature', axis= 1)

    x2 = data2.drop('Kvalitet', axis = 1)
    x2 = x2.drop('Unnamed: 5', axis = 1)
    # x2 = x2.drop('Till Datum Tid (UTC)', axis = 1)
    x2 = x2.drop('Fran Datum Tid (UTC)', axis = 1)
    x2 = x2.drop('Tidsutsnitt:', axis = 1)
    y2 = x2.temperature
    X2 = x2.drop('temperature', axis= 1)

    new_dates = []
    counter = 0
    X = X.append(X2)
    dates = X.day
    for day in dates:
        day = datetime.datetime.strptime(day, "%Y-%m-%d")
        day2 = (day - datetime.datetime(1970,1,1)).total_seconds()
        new_dates.append(day2)
    X.day = new_dates
    new_dates= []
    for day in X.till:
        day = datetime.datetime.strptime(day, "%Y-%m-%d %H:%M:%S")
        day2 = (day - datetime.datetime(1970,1,1)).total_seconds()
        new_dates.append(day2)
    X.till = new_dates
    y = y.append(y2)


    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5, 
                                                        random_state=123, 
                                                        )


    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
 
    pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

    hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1], }

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print r2_score(y_test, pred)
    print mean_squared_error(y_test, pred)

    joblib.dump(clf, 'weather_predictor.pkl')

def get_the_weather(date):
    weather = data1.day
    temp = data1.temperature

    for i in range(0, len(weather)):
        day = datetime.datetime.strptime(weather[i], "%Y-%m-%d")
        if (day == date):
            return temp[i]
            


def predict_weather():
    clf = joblib.load('weather_predictor.pkl')
    print("-" * 48)
    print("Enter the details of the date you would like to predict")
    print("\n")
    option = input("Year: ")
    year = option
    option = input("Month number (00): ")
    month = option
    option = input("Day number (00): ")
    theday = option

    day = str(year) + "-" + str(month) + "-" + str(theday)
    day = datetime.datetime.strptime(day, "%Y-%m-%d")
    date = (day - datetime.datetime(1970,1,1)).total_seconds()

    day_x = str(year) + "-" + str(month) + "-" + str(theday+1)
    day_x = datetime.datetime.strptime(day_x, "%Y-%m-%d")
    date_x = (day_x - datetime.datetime(1970,1,1)).total_seconds()

    X = [[date, date_x]]
    print("\n")
    print("-" * 48)
    print("The temperature is predicted to be: " + str(clf.predict(X)[0]))
    print("The temperature was actually: " + str(get_the_weather(day)))
    print("-" * 48)
    print("\n")

def run_menu():
    print("*" *48)
    print("-" *10 + " What would you like to do? " + "-" *10)
    print("\n")
    print("1. Look up the weather on a specific day")
    print("2. Predict the weather on a specific day")
    print("\n")

    option = input("Enter option: ")

    while True:
        if option == 2 or option == 1 or option == 9:
            break
        option = input("Enter option: ")
    return option

def run_program(option):
    if option == 1:
        print("1")
    elif option == 2:
        predict_weather()

if __name__== "__main__":
    train_data()

    while True:
        option = run_menu()
        if option == 9:
            break
        else:
            run_program(option)



