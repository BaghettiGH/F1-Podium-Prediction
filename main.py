#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
#######################
# Page configuration
st.set_page_config(
    page_title="F1 Podium Prediction", # Replace this with your Project's Title
    page_icon="assets/f1logo.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    st.image("assets/f1logo.png",use_column_width="auto")
    # Sidebar Title (Change this with your project's title)
    st.title('F1 Podium Prediction')
    
    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Group 7 Members")
    st.markdown("1. Dizon, Ma.Sophia\n2. Hojilla, Guillan\n3. Jaso, Louis Patrick\n4. Molina, Patrick Lawrence\n3. Nanwani, Pratik")

#######################
# Data

# Load data
df_drivers = pd.read_csv('data/drivers.csv')
df_qualifying = pd.read_csv('data/qualifying.csv', na_values=['\\N',''])
df_races = pd.read_csv('data/races.csv')
df_results = pd.read_csv('data/results.csv')

df_driver_results = pd.merge(df_results,df_drivers[['driverId','driverRef']], on='driverId',how='inner') #merging driver.csv & results.csv
df_driver_race_results= pd.merge(df_driver_results,df_races[['raceId','name','year']], on='raceId', how='inner') #merging df_driver_results & races.csv
df_drivers_quali_race_results = pd.merge(df_driver_race_results,df_qualifying[['raceId','driverId','q1','q2','q3']], on=['raceId','driverId'], how='inner') # merging df_driver_race_results & qualifying.csv
main_column = ['driverRef','name','year','q1','q2','q3','grid','positionOrder'] #selecting columns
df_main = df_drivers_quali_race_results[main_column]
df_main.rename(columns={'driverRef':'driver_name',
                        'name':'race_name',
                        'year':'race_year',
                        'grid':'starting_grid_position',
                        'positionOrder':'finishing_position'},inplace=True) #renaming columsn
df_main= df_main.sort_values(by='race_year',ascending=False) #sorting by race_year
df_main.dropna(subset=['q1'], inplace=True)
df_main = df_main.fillna(0)

def timetoseconds(time_str): # Turning q1,q2,q3 times to seconds format
  time_str = str(time_str)
  if time_str == "0" or pd.isna(time_str):
    return 0
  try:
    minutes,seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)
  except ValueError:
    return 0
  
df_main['q1_seconds'] = df_main['q1'].apply(timetoseconds)
df_main['q2_seconds'] = df_main['q2'].apply(timetoseconds)
df_main['q3_seconds'] = df_main['q3'].apply(timetoseconds)
df_main['avg_qualifying_time'] = df_main[['q1_seconds', 'q2_seconds', 'q3_seconds']].mean(axis=1)

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("**Formula 1 World Championship (1950 - 2024)** is a dataset made by a user in Kaggle named *Vopani*. It contains 14 csv files ranging from `circuits`, `lap_times`, `pit_stops`, `drivers`, and more. However, in this application we will only use  `drivers.csv`, `qualifying.csv`, `races.csv`, `results.csv` to predict drivers on the podium.") #show each dataset here 
    st.markdown(""" 
                - The `driver.csv` contains Formula 1 Drivers basic information from 1950 - 2024.  
                - The `qualifying.csv` contains the times made by drivers in qualifying sessions.  
                - The `races.csv` contains the year, race name, and other information of each races in Formula 1.  
                - The `results.csv` contains information about the initial position of drivers and their final position in the race.  
                """)
    st.write("To make a prediction, we will only use multiple features from the csvs such as:")
    st.markdown("""  
                - drivers' name, 
                - race name, 
                - race year, 
                - q1, q2, q3 times,  
                - initial position,  
                - finishing position,  
                - average qualifying time
                 """)
    st.subheader("Formula 1 Main Dataframe")
    st.dataframe(df_main,use_container_width = True, hide_index=True)
    st.subheader("Descriptive Statistics")
    st.dataframe(df_main.describe(),use_container_width = True)
    st.markdown(""" 
                - The average starting grid position is **10.9** and standard deviation of **6.24** which has a high variation.
                - The average finishing position is **11.203** and standard deviation of *6.291** which has a high variation.
                - The q1 average time is **88.531** with a standard deviation of **15.467** which has moderate variation.  
                - The q2 average time is **49.269** with a standard deviation of **44.497** which has a high variation.
                - The q3 average time is **30.413** with a standard deviation of **42.262** which has extremely high variation.
                - The mean average qualifying time is **56.071** with a standard deviation of **27.107** which has has a moderately high variation
                """)
    # Having a high variation in a dataset means that the model will have a hard time seeing patterns.
    # Recommendation: Handle outliers, normalize the data. 
    
    # Your content for your DATASET page goes here

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")


    col = st.columns((1.5, 4.5, 2), gap='medium')

    # Your content for the EDA page goes here

    with col[0]:
        st.markdown('#### Graphs Column 1')


    with col[1]:
        st.markdown('#### Graphs Column 2')
        
    with col[2]:
        st.markdown('#### Graphs Column 3')

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    # buffer = io.StringIO()
    # df_main.info(buf=buffer)
    # info_str = buffer.getvalue()
    # st.text(info_str)
    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here