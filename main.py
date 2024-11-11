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
df_main_dirty = df_drivers_quali_race_results[main_column]
df_main_dirty.rename(columns={'driverRef':'driver_name',
                        'name':'race_name',
                        'year':'race_year',
                        'grid':'starting_grid_position',
                        'positionOrder':'finishing_position'},inplace=True) #renaming columsn
df_main= df_main_dirty.sort_values(by='race_year',ascending=False) #sorting by race_year
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

######## For Model Training
df_model = df_main
encoder = LabelEncoder()
df_model['driver_Encoded'] = encoder.fit_transform(df_model['driver_name'])
df_model['race_Encoded'] = encoder.fit_transform(df_model['race_name'])

df_model['top_finish'] = df_main['finishing_position'].apply(lambda x:1 if x<=3 else 0)
features = ['avg_qualifying_time','starting_grid_position','driver_Encoded']
X = df_model[features]
Y = df_model['top_finish']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


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

    col = st.columns((1.5, 1.5, 1.5,1.5), gap='medium')
    with col[0]:
        st.markdown("drivers.csv")
        st.dataframe(df_drivers,hide_index=True)
    with col[1]:
        st.markdown("qualifying.csv")
        st.dataframe(df_qualifying,hide_index=True)
    with col[2]:
        st.markdown("results.csv")
        st.dataframe(df_results,hide_index=True)
    with col[3]:
        st.markdown("races.csv")
        st.dataframe(df_races,hide_index=True)
    st.markdown("Since the csv files has primary keys and are in different csvs, merge the datasets using **inner join**")
    st.markdown("### Merging Datasets")
    st.code("""
            df_driver_results = pd.merge(df_results,df_drivers[['driverId','driverRef']], on='driverId',how='inner') #merging driver.csv & results.csv
            df_driver_race_results= pd.merge(df_driver_results,df_races[['raceId','name','year']], on='raceId', how='inner') #merging df_driver_results & races.csv
            df_drivers_quali_race_results = pd.merge(df_driver_race_results,df_qualifying[['raceId','driverId','q1','q2','q3']], on=['raceId','driverId'], how='inner') # merging df_driver_race_results & qualifying.csv        
            """)
    st.markdown("### Initializing df_main")
    st.code("""
            main_column = ['driverRef','name','year','q1','q2','q3','grid','positionOrder'] #selecting columns
            df_main = df_drivers_quali_race_results[main_column]
            df_main.rename(columns={'driverRef':'driver_name',
                        'name':'race_name',
                        'year':'race_year',
                        'grid':'starting_grid_position',
                        'positionOrder':'finishing_position'},inplace=True) #renaming column
            """)
    col1 = st.columns((2.5,1.5), gap='medium')

    with col1[0]:
        st.dataframe(df_main_dirty,hide_index = True)
    with col1[1]:
        st.markdown("#### Check for null values")
        st.dataframe(df_main_dirty.isnull().sum(),use_container_width = True)
    st.write("df_main still has null values and qualifying times are unusable.")
    st.markdown("### Data Cleaning")
    st.code("""
            df_main.dropna(subset=['q1'], inplace=True)
            df_main = df_main.fillna(0)
            """)
    st.code("""
            def timetoseconds(time_str): # Turning q1,q2,q3 times to seconds format
                time_str = str(time_str)
                if time_str == "0" or pd.isna(time_str):
                    return 0
                try:
                    minutes,seconds = time_str.split(':')
                    return int(minutes) * 60 + float(seconds)
                except ValueError:
                    return 0
            """)
    st.code("""
            df_main['q1_seconds'] = df_main['q1'].apply(timetoseconds)
            df_main['q2_seconds'] = df_main['q2'].apply(timetoseconds)
            df_main['q3_seconds'] = df_main['q3'].apply(timetoseconds)
            df_main['avg_qualifying_time'] = df_main[['q1_seconds', 'q2_seconds', 'q3_seconds']].mean(axis=1)  
            """)
    st.dataframe(df_main,hide_index = True)
    buffer = io.StringIO()
    df_main.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    st.markdown("### Encoding Labels")
    st.code("""
            encoder = LabelEncoder()
            df_model['driver_Encoded'] = encoder.fit_transform(df_model['driver_name'])
            df_model['race_Encoded'] = encoder.fit_transform(df_model['race_name'])
            df_model['top_finish'] = df_main['finishing_position'].apply(lambda x:1 if x<=3 else 0)
            """)
    st.markdown("### Train-Test Split")
    st.code("""
            features = ['avg_qualifying_time','starting_grid_position','driver_Encoded']
            X = df_model[features]
            Y = df_model['top_finish']
            """)
    st.code("""
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            """)
    col1 = st.columns((3.5, 3.5), gap='medium')

    # Your content for the EDA page goes here

    with col1[0]:
        st.markdown("#### X_train")
        st.dataframe(X_train,hide_index = True,use_container_width = True)


    with col1[1]:
        st.markdown("#### X_test")
        st.dataframe(X_test, hide_index = True,use_container_width = True)
    col2 = st.columns((2.5,2.5),gap='medium')
    with col2[0]:
        st.markdown("#### Y_train")
        st.dataframe(Y_train,hide_index = True,use_container_width = True)

    with col2[1]:
        st.markdown("#### Y_test")
        st.dataframe(Y_test, hide_index = True,use_container_width = True)
    # Your content for the DATA CLEANING / PREPROCESSING page goes here

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Your content for the MACHINE LEARNING page goes here
Model Training

df_model = df_main
df_model

encoder = LabelEncoder()
df_model['driver_Encoded'] = encoder.fit_transform(df_model['driver_name'])
df_model['race_Encoded'] = encoder.fit_transform(df_model['race_name'])

df_model['top_finish'] = df_main['finishing_position'].apply(lambda x:1 if x<=3 else 0)

df_model

features = ['avg_qualifying_time','starting_grid_position','driver_Encoded']
X = df_model[features]
Y = df_model['top_finish']

print("X.shape:", (X.shape))
print("Y.shape:", (Y.shape))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train,Y_train)

Model Evaluation

predict_train = model.predict(X_train)

accuracy_train = accuracy_score(predict_train,Y_train)
print(f'Accuracy: {accuracy_train * 100:.2f}%')

predict_test = model.predict(X_test)
accuracy_test = accuracy_score(predict_test,Y_test)
print(f'Accuracy: {accuracy_test * 100:.2f}%')

classification_rep = classification_report(Y_test, predict_test)
classification_rep

importance = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance:':importance
})

importance_df

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Your content for the PREDICTION page goes here
predict_train = model.predict(x_train)
accuracy_train = accuracy_store(predict_train, Y_train)
print(f'Accuracy: {accuracy_train * 100:.2f}%')

predict_test = model.predict(x_test)
accuracy_test = accuracy_score(predict_test, Y_test)
print(f'Accuracy: {accuracy_test * 100:.2f}%')

classification_rep = classification_report(Y_test, predict_test)
classification_rep

importance = model.coef_[0]
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance:':importance
})

importance_df

def translate_input(q1_time, q2_time, q3_time, grid_position, driver_name):
    q1_time = timetoseconds(q1_time)
    q2_time = timetoseconds(q2_time)
    q3_time = timetoseconds(q3_time)

avg_q_time = (q1_time + q2_time + q3_time_/3
              driver_name = driver_name.lower()
              driver_name_mapping = df_model[['driver_Encoded', 'driver_name']].drop_duplicates().sort_values('driver_Encoded')
              driver_label = driver_name_mapping.loc[driver_name_mapping['driver_name']==driver_name,'driver_Encoded'].values[0]
              input_data = [avg_q_time, grid_position, driver_label]
              predict_instance(input_data, driver_name)
                     return input_data

def get_prediction_input():
 lap1 = input("Enter Lap 1 time (format 'm:ss.sss'): ")
 lap2 = input("Enter Lap 2 time (format 'm:ss.sss'): ")
 lap3 = input("Enter Lap 3 time (format 'm:ss.sss'): ")
 position = int(input("Enter starting position: "))
 driver = input("Enter driver name: ").lower()
    
    input_data = translate_input(lap1, lap2, lap3, position, driver)
    input_data = np.array(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data)
    return prediction

    pred_instance = get_prediction_input()
    print("Prediction:", pred_instance)
        
# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    # Your content for the CONCLUSION page goes here
