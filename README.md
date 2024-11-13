# F1 Podium Prediction
A Streamlit web application that predicts if a driver will finish on the podium (1st,2nd, or 3rd Place) in a Formula 1 Grand Prix Race using their qualifying times and initial position. This application performs **EDA**, **Data Preprocessing**, and **Supervised Machine Learning** to predict instances using **Logistic Regression**.


![Main Page Screenshot](screenshots/main_page_screenshot.png)



### 🔗 Links:  
-  [Streamlit Link](https://f1-podium-prediction.streamlit.app/)
-  [Google Colab Notebook](https://colab.research.google.com/drive/1AxRBCJX24u00DtShTovHU3tbSkpzfIt7?usp=sharing)


### 📊 Dataset:
- [Formula 1 World Championship (1950 - 2024)(Kaggle)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
This repository serves as a project guide template for my students in **Introduction to Data Science** course for their final project. It contains a Python file `dashboard_template.py` which contains a boilerplate for a Streamlit dashboard.
### 📖 Pages:

1. `Dataset` - Brief description of the Formula 1 Dataset used in this dashboard.
2. `EDA` - Exploratory Data Analysis of the F1 Dataset. Highlights relationship of initial positionl, qualifying times and finishing on the podium. Includes bar graphs,histogram, scatter graph, etc.
3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps such as encoding the species column and splitting the dataset into training and testing sets.
4. `Machine Learning` - Training Logistic Regression model. This page also includes the model evaluation, feature importance and classification report.
5. `Prediction` - Prediction page where users can input values to predict if the driver will finish on the podium.
6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

### 💡 Findings / Insights
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
From this data science activity, it is evident that the dataset needs more cleaning as there is still a high variation in some features used in the model.
Regardless of the improvements needed, the students were able to display and see correlation between variables using data visualization.
Moreover, the students have also managed make an F1 Podium Predictor based on user inputs           
