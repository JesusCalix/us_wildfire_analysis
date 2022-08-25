from app import app
from flask import render_template

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func, asc, desc

from ..models import Fire

import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def get_year_count(session, ascending=False):
    direction = asc if ascending else desc
    return session.query(
            Fire.fire_year, func.count(Fire.objectid).label("total_fires")
         ).group_by(Fire.fire_year)

def get_count_by_county(session,ascending=False):
    direction = asc if ascending else desc
    return session.query(
            Fire.fips_name, func.count(Fire.objectid).label("total_fires")
         ).filter(Fire.fips_name != None).group_by(Fire.fips_name).order_by(direction('total_fires')).limit(15)

def get_features(session,ascending=False):
    direction = asc if ascending else desc
    return session.query(
            Fire.fire_size,Fire.latitude,Fire.longitude,Fire.discovery_date,Fire.fire_year,Fire.stat_cause_code, Fire.stat_cause_descr
         )
    
def create_rf_model(session,query):
    df = pd.read_sql(query.statement, query.session.bind)
    # print(df.isna().sum())
    # print(df[['stat_cause_code','stat_cause_descr']].value_counts())
    df['date'] = pd.to_datetime(df.loc[:,'discovery_date'] - pd.Timestamp(0).to_julian_date(), unit='D')
    df['month'] = pd.DatetimeIndex(df.loc[:,'date']).month

    df["stat_cause_code"] = df.apply(lambda x: 1 if x["stat_cause_code"] == 1 else x["stat_cause_code"],axis=1)
    df["stat_cause_code"] = df.apply(lambda x: 2 if x["stat_cause_code"] in [5,2,6,11,12] else x["stat_cause_code"],axis=1)
    df["stat_cause_code"] = df.apply(lambda x: 3 if x["stat_cause_code"] in [7,4,8,3,10] else x["stat_cause_code"],axis=1)
    df["stat_cause_code"] = df.apply(lambda x: 4 if x["stat_cause_code"] in [9,13] else x["stat_cause_code"],axis=1)

    prediction_data = df[['fire_size', 'latitude', 'longitude', 'fire_year','stat_cause_code', 'month']].copy()

    # Split into x and y
    x = prediction_data.drop(['stat_cause_code'], axis = 1).values
    y = prediction_data['stat_cause_code'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

    rf = RandomForestClassifier(random_state=0, n_estimators = 10, oob_score = True)
    rf = rf.fit(x_train, y_train)

    # Make predictions for the test set
    y_pred = rf.predict(x_test)

    labels=['Nature','Accident','Human','Other']
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    return accuracy_score(y_test, y_pred), matrix, labels


def create_session():
    engine = create_engine('sqlite:///./data/FPA_FOD_20170508.sqlite')
    connection = engine.connect()

    engine = create_engine('sqlite:///./data/FPA_FOD_20170508.sqlite')
    Session = sessionmaker()
    Session.configure(bind=engine)
    session = Session()

    return session


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/wildfires_trend", methods=['GET', 'POST'])
def wildfires_trend():
    session = create_session()
    result = get_year_count(session)
    session.close()

    labels = []
    values = []

    for row in result:
        # print(f"{row.fire_year},{row.total_fires}")
        labels.append(row.fire_year)
        values.append(row.total_fires)

    return render_template("wildfires_trend.html",result=result,max=max(values)+10000,labels=labels, values=values)

@app.route("/fire_prone_counties", methods=['GET', 'POST'])
def fire_prone_counties():
    session = create_session()
    result = get_count_by_county(session,ascending=False)
    result_2 = get_count_by_county(session,ascending=True)
    session.close()

    labels = []
    values = []

    for row in result:
        # print(f"{row.fire_year},{row.total_fires}")
        labels.append(row.fips_name)
        values.append(row.total_fires)

    labels_2 = []
    values_2 = []

    for row in result_2:
        # print(f"{row.fire_year},{row.total_fires}")
        labels_2.append(row.fips_name)
        values_2.append(row.total_fires)

    return render_template("fire_prone_counties.html",result=result,max=max(values)+10000,labels=labels, values=values,result_2=result_2,max_2=max(values_2),labels_2=labels_2, values_2=values_2)

@app.route("/prediction_analysis", methods=['GET', 'POST'])
def prediction_analysis():
    # return render_template("fire_prone_counties.html",result=result,max=max(values)+10000,labels=labels, values=values)
    # return render_template("prediction_analysis.html",accuracy=accuracy,matrix=matrix,labels=labels)
    return render_template("prediction_analysis.html",visibility="hidden" )


@app.route('/prediction',methods=['POST'])
def prediction():
    session = create_session()
    result = get_features(session)
    session.close()

    accuracy,matrix,labels = create_rf_model(session,result)

    print(accuracy)
    print(matrix)

    return render_template('prediction_analysis.html',accuracy=accuracy,matrix=matrix,labels=labels)