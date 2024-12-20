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

    st.markdown("""
        A Streamlit web application allows users to interactively analyze and visualize data. 
    Users can upload files or input parameters, and the app processes this data to generate insights through visualizations like charts and graphs. 
    It might also run machine learning models, providing predictions or evaluating performance in real time. 
    Streamlit makes complex data tasks accessible and easy to explore for users.
                """)
    st.markdown("""
        In Formula 1, qualifying determines starting positions for the race. It’s split into three sessions:

        - **Q1**: All drivers participate, with the five slowest eliminated and starting from positions 16-20.
        - **Q2**: The remaining 15 compete, with five more eliminated for positions 11-15.
        - **Q3**: The fastest 10 fight for the top 10 grid spots, with the quickest taking pole position.
    """)    
    col8 = st.columns((1.5,4.5,1.5))
    with col8[1]:
        st.image("assets/f1_pic.jpg",width=800,caption='Formula 1')


    st.markdown("""
    In some races, a sprint qualifying format determines the grid through a shorter race on Saturday.
    Starting positions are crucial for podium predictions, as front-runners have better chances due to fewer overtakes needed.
    ### Pages

    `Dataset` - Brief description of the Formula 1 Dataset used in this dashboard.  
    `EDA` - Exploratory Data Analysis of the F1 Dataset. Highlights relationship of initial position, qualifying times and finishing on the podium. Includes bar graphs,histogram, scatter graph, etc.  
    `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.  
    `Machine Learning` - Training Logistic Regression model. This page also includes the model evaluation, feature importance and classification report.  
    `Prediction` - Prediction page where users can input values to predict if the driver will finish on the podium.  
    `Conclusion` - Summary of the insights and observations from the EDA and model training.    
        """)
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
  
    # Your content for the EDA page goes here   
    col = st.columns((3.5,4.5),gap = 'medium')
    col6 = st.columns((1.5,4.5,1.5))
    with col[0]:
        filtered_df = df_main[(df_main['starting_grid_position'] > 0) & (df_main['finishing_position'] > 0)]

        fig, ax = plt.subplots(figsize =(10,6))
        ax.scatter(filtered_df['starting_grid_position'], filtered_df['finishing_position'], alpha=0.1)
        plt.title('Scatter Plot: Starting Position vs. Finishing Position')
        plt.xlabel('Starting Grid Position')
        plt.ylabel('Finishing Position')
        plt.gca().invert_yaxis()
        plt.grid(True)
        st.pyplot(fig)
        
        st.write("In the scatter plot, you may observe that drivers starting from the front tend to have better finishing positions.")

    
        fig,ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_main['avg_qualifying_time'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Drivers' Average Qualifying Times")
        plt.xlabel("Average Qualifying Time (seconds)")
        plt.ylabel("Frequency of Drivers")
        plt.grid(axis='y', alpha=0.75)
        st.pyplot(fig)
        
        st.write('The histogram will display the frequency distribution of average qualifying times, helping to identify any central tendencies or spread in qualifying times across drivers. Peaks or clusters may indicate typical qualifying performance ranges.')
    with col[1]:        
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return None
            time_str = time_str.strip()
            minutes, seconds = time_str.split(':')
            return int(minutes) * 60 + float(seconds)

        df_qualifying['q1_seconds'] = df_qualifying['q1'].apply(time_to_seconds)
        df_qualifying['q2_seconds'] = df_qualifying['q2'].apply(time_to_seconds)
        df_qualifying['q3_seconds'] = df_qualifying['q3'].apply(time_to_seconds)

        average_q1_per_driver = df_qualifying.groupby('driverId')['q1_seconds'].mean().dropna()
        average_q2_per_driver = df_qualifying.groupby('driverId')['q2_seconds'].mean().dropna()
        average_q3_per_driver = df_qualifying.groupby('driverId')['q3_seconds'].mean().dropna()

        fig,ax = plt.subplots(1, 3, figsize=(18, 6))

        # Q1 Histogram
        ax[0].hist(average_q1_per_driver, bins=20, color='green', edgecolor='black')
        plt.xlabel('Average Q1 Time (seconds)')
        plt.ylabel('Number of Drivers')
        plt.title('Average Q1 Time')

        # Q2 Histogram
        ax[1].hist(average_q2_per_driver, bins=20, color='green', edgecolor='black')
        plt.xlabel('Average Q2 Time (seconds)')
        plt.title('Average Q2 Time')

        # Q3 Histogram
        ax[2].hist(average_q3_per_driver, bins=20, color='purple', edgecolor='black')
        plt.xlabel('Average Q3 Time (seconds)')
        plt.title('Average Q3 Time')

        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("Based off of the 3 histograms, we can see that in each qualifying session, the most common time reached is just around the 90 second mark.")
        
        def time_to_seconds(time_str):
            if pd.isna(time_str):
                return None
            time_str = time_str.strip()
            minutes, seconds = time_str.split(':')
            return int(minutes) * 60 + float(seconds)

        df_qualifying['q1_seconds'] = df_qualifying['q1'].apply(time_to_seconds)
        df_qualifying['q2_seconds'] = df_qualifying['q2'].apply(time_to_seconds)
        df_qualifying['q3_seconds'] = df_qualifying['q3'].apply(time_to_seconds)

        avg_qualifying_times = df_qualifying.groupby('driverId')[['q1_seconds', 'q2_seconds', 'q3_seconds']].mean().dropna().reset_index()
        merged_data = pd.merge(avg_qualifying_times, df_results, on='driverId')
        fig, axs = plt.subplots(1, 3, figsize=(14, 8))

        # Q1 Boxplot
        axs[0].boxplot(merged_data['q1_seconds'], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        plt.xlabel('Average Q1 Time (seconds)')
        plt.ylabel('Number of Wins')
        plt.title('Wins vs. Average Q1 Time')

        # Q2 Boxplot

        axs[1].boxplot(merged_data['q2_seconds'], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        plt.xlabel('Average Q2 Time (seconds)')
        plt.title('Wins vs. Average Q2 Time')

        # Q3 Boxplot

        axs[2].boxplot(merged_data['q3_seconds'], patch_artist=True, boxprops=dict(facecolor="lightgreen"))
        plt.xlabel('Average Q3 Time (seconds)')
        plt.title('Wins vs. Average Q3 Time')

        plt.tight_layout()
        st.pyplot(fig)
        st.write("Based on the box plot, average number of wins for each average Q1/Q2/Q3 time are different but they have a common midpoint of around 88 or 87.")
    with col6[1]:    
        fig,ax = plt.subplots(figsize =(8,6))
        corr_matrix = df_main[['starting_grid_position', 'avg_qualifying_time', 'finishing_position']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig, use_container_width = True)
        st.markdown(""" 
                From the matrix, we can observe that there is a high positive correlation between finishing_position and starting_grid_position. This means that when the driver starts in higher places such as 14th place. The driver will finish with high finishing position.
                Moreover, we can also observe that there is a negative correlation between starting_grid_position and avg_qualifying_time. This means that when their average qualifying time is high. The driver's initial position for the race tends to be low.
                Lastly, there is also negative correlation between avg_qualifying_time and finishing_position. This means that when they finished the race in the last place. The driver has a low avg_qualifying_time.
                """)
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

elif st.session_state.page_selection == "machine_learning":
    # Machine Learning Page
    st.header("🏎️ Machine Learning")

    # Brief explanation of Logistic Regression
    st.subheader("Why use Logistic Regression for Predicting F1 Podium?")
    st.markdown("""
    Logistic Regression is a suitable choice for this task because it's a classification algorithm, which can help predict categorical outcomes—in this case, whether a driver finishes on the podium or not.

    ### How Logistic Regression Works
    Logistic Regression estimates the probability of a binary event by fitting data to a logistic function. It’s commonly used for binary classification tasks and can be extended to multiclass classification as well.
    """)
    # Simulate dataset
    st.subheader("Dataset Preparation")
    df_main = df_model
    st.dataframe(df_model.head(),hide_index = True)

    # Encode categorical variables
    st.subheader("Data Preprocessing")
    st.markdown("Encoding categorical variables")

    # Encoding driver_name, race_name, and finishing_position
    encoder = LabelEncoder()
    df_model['driver_Encoded'] = encoder.fit_transform(df_model['driver_name'])
    df_model['race_Encoded'] = encoder.fit_transform(df_model['race_name'])

    df_driver_encoded=df_model[['driver_name', 'driver_Encoded']].drop_duplicates().sort_values('driver_Encoded').reset_index(drop=True)
    df_race_encoded = df_model[['race_name','race_Encoded']].drop_duplicates().sort_values('race_Encoded').reset_index(drop = True)

    # Display encoded DataFrame
    col3 = st.columns((3.5, 3.5), gap='medium')
    with col3[0]:
        st.dataframe(df_driver_encoded,hide_index = True,use_container_width = True)
    with col3[1]:
        st.dataframe(df_race_encoded, hide_index = True,use_container_width = True)

    st.subheader("Training the Logistic Regression Model")
    st.code("""
            log_reg = LogisticRegression()
            log_reg.fit(X_train, Y_train)
            """)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    col4 = st.columns((3.5, 3.5), gap='medium')
    with col4[0]:
        # Model Evaluation
        st.subheader("Model Evaluation")
        # Accuracy on train data
        train_accuracy = log_reg.score(X_train, Y_train)
        st.write(f"Train Accuracy: {train_accuracy * 100:.2f}%")

        # Accuracy on test data
        y_pred = log_reg.predict(X_test)
        test_accuracy = accuracy_score(Y_test, y_pred)
        st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    with col4[1]:
    # Classification Report
        st.write("Classification Report:")
        st.text(classification_report(Y_test, y_pred))
        st.write("The model seems to perform better when predicting losses compared to predicting wins. This may suggest that class imbalance is still present in the dataset.")
    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.Series(log_reg.coef_[0], index=X.columns)
    st.write(feature_importance)
    st.write('The `starting_grid_position` has a high magnitude and negative correlation in affecting the prediction of the model. This means that there is a high chance of being on the podium if the driver has a low `starting_grid_position`.')


# Prediction Page

elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    log_reg = LogisticRegression()
    log_reg.fit(X_train, Y_train)
    driver_images = {
        "albon": "assets/albon.png",
        "alonso": "assets/alonso.png",
        "bearman": "assets/bearman.png",
        "gasly": "assets/gasly.png",
        "hamilton": "assets/hamilton.png",
        "hulkenberg": "assets/hulkenberg.png",
        "kevin_magnussen": "assets/kevin_magnussen.png",
        "lawson": "assets/lawson.png",
        "leclerc": "assets/leclerc.png",
        "max_verstappen": "assets/max_verstappen.png",
        "norris": "assets/norris.png",
        "ocon": "assets/ocon.png",
        "perez": "assets/perez.png",
        "piastri": "assets/piastri.png",
        "russel": "assets/russel.png",
        "sainz": "assets/sainz.png",
        "stroll": "assets/stroll.png",
        "tsunoda": "assets/tsunoda.png",
    }


    col_pred = st.columns((1.5, 3), gap='medium')
    def driver_img(driverName):
        if driverName in driver_images:
            st.image(driver_images[driverName],width=350)
        else:
            return None
    def translate_input(q1_time,q2_time,q3_time,grid_position,driver_name):
        q1_time= timetoseconds(q1_time)
        q2_time= timetoseconds(q2_time)
        q3_time= timetoseconds(q3_time)
        avg_q_time = (q1_time+q2_time+q3_time)/3
        driver_name = driver_name.lower()
        driver_name_mapping = df_model[['driver_Encoded', 'driver_name']].drop_duplicates().sort_values('driver_Encoded')
        if driver_name not in driver_name_mapping['driver_name'].values:
            st.warning(f"Driver name: '{driver_name} cannot be found")
            return None
        driver_label = driver_name_mapping.loc[driver_name_mapping['driver_name']==driver_name,'driver_Encoded'].values[0]
        input_data = [avg_q_time,grid_position,driver_label]
        predict_instance(input_data,driver_name)
        
        
        return input_data
    def predict_instance(input_data,name):
        input_data = np.array(input_data).reshape(1,-1)
        pred_instance = log_reg.predict(input_data)
        driver_img(name )
        if(pred_instance [0]==1):
            st.markdown(f"**{name.capitalize()}** has a high chance of finishing on the podium!")
        else:
            st.markdown(f"**{name.capitalize()}** has a low chance of finishing on the podium.")

    if 'clear' not in st.session_state:
        st.session_state.clear = False


    with col_pred[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:
                st.session_state.clear = True
                q1_time=0
                q2_time=0
                q3_time=0

    with col_pred[1]:
        st.markdown("#### 🚗 F1 Result Predictor")

        q1_time = st.text_input('Q1 Lap Time (format "m:ss.sss")', placeholder='0:00.000')
        q2_time = st.text_input('Q2 Lap Time (format "m:ss.sss")', placeholder='0:00.000')
        q3_time = st.text_input('Q3 Lap Time (format "m:ss.sss")', placeholder='0:00.000')
        grid_position = st.number_input('Starting Grid Position', min_value=1, max_value=20, step=1)
        driver_name = st.text_input('Driver Name').lower()

        if st.button('Predict Podium'):
            translate_input(q1_time,q2_time,q3_time,grid_position,driver_name)
            

              

    if show_dataset:
        st.subheader("Dataset")
        st.dataframe(df_model, use_container_width=True,hide_index=True)









# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    # Your content for the CONCLUSION page goes here
    st.markdown("""
                
    Through exploratory data analysis and training of `Logistic Regression` on the **Formula 1 dataset**, the key insights and observations are:

    #### 1. 📊 **Dataset Characteristics**:
                
    - The average `starting_grid_position` and average `finishing_position` has high variation which could impact model performance negatively.
    - The average q3 time has extremely high variation.
    - Further data cleaning needs to be performed such as managing outliers.
    #### 2. 📝 **Feature Distributions and Separability**:
                
    - Starting from the front tend to have better finishing positions.
    - The most common lap time obtained in qualifying sessions is around **90** seconds
    - Average qualifying lap time is under 100 seconds.
    - Drivers with faster Q3 times have more wins.
    - There is a high positive correlation between `finishing_position` and `starting_grid_position`.
    - There is a moderate negative correlation between `avg_qualifying_time` and `finishing_position`.
                
    #### 3. 📈 **Model Performance (Decision Tree Classifier)**:
            
    - The model acquired an accuracy of **89.33%** for train set.
    - The model acquired an accuracy of **87.33%** for test set.
    - Precision in classifying loss is **91%**.
    - Precision in classifying win is **60%**.
    - Recall for loss is **95%**.
    - Recall for win  is **44%**.
    - F1-Score for loss is **93%**.
    - F1-Score for win is **51%**.
    - Feature Importance: 
        - avg_qualifying_time: 0.0036
        - starting_grid_position: -0.4116
        - driver_Encoded: -0.0023
    - From the results above, the model performs better when determining if the driver will not be on the podium. The models seems to be biased towards predicting losses, this means that there is class imbalance in the dataset.
    - The `starting_grid_position` appears to be the most important feature when predicting a win for the driver. This suggests that a lower `starting_grid_position` increases the chances of winning.

    ##### **Summing up:**  
    From this data science activity, it is evident that the dataset needs more cleaning as there is still a high variation in some features used in the model.
    Despite having a **87.33%** model accuracy, the model seems to be biased towards predicting losses and still needs improvement on predicting podiums which is the objective of the ML model.
    Some recommendations to improve model performance are: managing class imbalance, outliers, and adding more features that are relevant in predicting a podium win.
    Regardless of the improvements needed, the students were able to display and see correlation between variables using data visualization.
    Moreover, the students have also managed make an F1 Podium Predictor based on user inputs           
    """)
