 User Activity Analysis & Machine Learning Project

## Project Overview

This project analyzes a publicly available **user activity dataset** from Kaggle and applies machine learning techniques to predict whether a user account is verified based on usage behavior and profile information.

The project demonstrates an end-to-end data science workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model building, evaluation, and comparison.



##  Objective

* Analyze user activity patterns
* Perform exploratory data analysis with visual insights
* Build and compare multiple machine learning models
* Predict whether a user account is verified



##  Dataset

* Source: Kaggle (Social Media Users dataset)
* Type:User activity / behavioral data
* Rows: ~10,000
* Columns: 7

### Key Features:

* Platform
* Owner
* Primary Usage
* Country
* Daily Time Spent (minutes)
* Date Joined
* Verified Account (Target Variable)



##  Tools & Technologies

* Python
* Pandas – data loading & manipulation
* NumPy – numerical operations
* Matplotlib & Seaborn – data visualization
* Scikit-learn – machine learning models & evaluation



##  Workflow

### 1️ Data Loading & Cleaning

* Loaded dataset from a local CSV file
* Converted date columns to datetime format
* Engineered a new feature: 'Account_Age_Days'
* Encoded categorical variables
* Verified and handled missing values
<img width="611" height="231" alt="Screenshot 2025-12-18 102915" src="https://github.com/user-attachments/assets/9d48f75c-1cd2-46bf-b31f-24ee2dc5c2b3" />




### 2️ Exploratory Data Analysis (EDA)

Performed EDA using 4 visualizations:

1. Verified vs Non-Verified Users (Bar Chart)
2. Daily Time Spent vs Verification Status (Box Plot)
3. Account Age vs Verification Status (Box Plot)
4. Correlation Heatmap
<img width="798" height="686" alt="Screenshot 2025-12-17 204009" src="https://github.com/user-attachments/assets/54df8058-2036-4b87-9cf1-6b7be2bb3fca" />
<img width="789" height="684" alt="Screenshot 2025-12-17 204017" src="https://github.com/user-attachments/assets/bf9d3590-1b2e-44e4-b2a1-51bd03a87d6e" />
<img width="795" height="686" alt="Screenshot 2025-12-17 204027" src="https://github.com/user-attachments/assets/3de200e2-57f6-49ac-9c95-36cbf69ad7fd" />
<img width="1255" height="844" alt="Screenshot 2025-12-17 204038" src="https://github.com/user-attachments/assets/88db08cf-d60f-431a-83ab-2ab1ef8ea84b" />


#### Key Insights:

* The dataset is balanced between verified and non-verified users
* Verified users tend to have slightly older accounts
* Usage time alone is not a strong predictor
* No strong linear correlations exist, indicating complex relationships



### 3️ Feature Engineering

* Created 'Account_Age_Days' from 'Date Joined'
* One-hot encoded categorical features
* Prepared data for machine learning models



## Machine Learning Models

Two classification models were built and compared:

###  Logistic Regression

* Used as a baseline linear model
* Required feature scaling using 'StandardScaler'
* Provides interpretability

###  Random Forest Classifier

* Non-linear ensemble model
* Handles complex feature interactions
* Does not require feature scaling


##  Model Evaluation

Models were evaluated using two metrics:

* Accuracy
* F1 Score

### Results Summary:

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Logistic Regression | Moderate | Moderate |
| Random Forest       | Higher   | Higher   |
<img width="366" height="210" alt="Screenshot 2025-12-18 102953" src="https://github.com/user-attachments/assets/15f1736e-1558-431a-a4fc-9a23800f913a" />

 Random Forest outperformed Logistic Regression, indicating that verification status depends on non-linear combinations of features.

---

##  Conclusion

* User verification cannot be predicted using a single dominant feature
* Combining multiple behavioral and profile features improves performance
* Ensemble models like Random Forest are more effective for this dataset

This project highlights practical skills in data analysis, visualization, machine learning, and model evaluation.



##  How to Run the Project

1. Clone the repository:

--git clone https://github.com/your-username/user-activity-ml-project.git


2. Install dependencies:

--pip install pandas numpy matplotlib seaborn scikit-learn


3. Run the script:

--python user_activity_analysis.py


##  Future Improvements

* Try additional models (XGBoost, Gradient Boosting)
* Perform hyperparameter tuning
* Add platform-specific analysis
* Deploy as a web app
  



