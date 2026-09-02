ReadME

## Machine Learning-Based Salary Range Prediction for Job Listings

### Project Overview

This project develops a machine learning solution to predict the minimum and maximum salary ranges associated with job listings using historical job-related information.

The solution aims to support:

- Recruitment planning
- Compensation budgeting
- Salary transparency
- Workforce planning
- Data-driven HR decision-making

---

## Problem Statement

Salary ranges can vary significantly depending on factors such as:

- Job title
- Job category
- Career level
- Agency
- Work location
- Division / work unit
- Employment type
- Job requirements
- Skills and responsibilities

Manually estimating an appropriate salary range can be difficult when dealing with a large number of job postings.

---

## Project Objectives

The main objectives of this project are to:

- Analyze historical job posting and salary patterns.
- Identify important attributes associated with salary variation.
- Perform data cleaning and preprocessing.
- Engineer relevant features for predictive modelling.
- Develop and compare multiple regression algorithms.
- Optimize the strongest models using hyperparameter tuning.
- Evaluate final models using MAE, RMSE and R².
- Analyze prediction errors and feature importance.
- Translate model findings into actionable HR recommendations.

---

## Dataset Overview

- **Total Records:** 5,120
- **Total Features:** 30

### Target Variables

| Target | Description |
|---|---|
| Salary Range From | Minimum salary offered |
| Salary Range To | Maximum salary offered |

### Major Feature Categories

| Category | Examples |
|---|---|
| Job Information | Business Title, Civil Service Title |
| Organization | Agency, Division / Work Unit |
| Job Classification | Job Category, Career Level, Level |
| Employment | Full-Time/Part-Time, # Of Positions |
| Location | Work Location, Work Location 1 |
| Job Content | Job Description, Preferred Skills |
| Requirements | Minimum Qual Requirements |
| Dates | Posting Date, Post Until, Posting Updated |

---

## Data Preparation

The dataset was examined for:

- Missing values
- Data types
- Duplicate records
- High-cardinality categorical variables
- Empty columns
- Irrelevant identifiers

`Recruitment Contact` contained no usable values and was identified for removal.

Missing values in text and categorical features were handled during preprocessing.

---

## Exploratory Data Analysis

EDA was performed to understand:

- Minimum salary distribution
- Maximum salary distribution
- Salary range variation
- Job category distribution
- Agency-level salary patterns
- Career level and salary relationship
- Location-based salary differences
- Categorical feature cardinality

The analysis showed that salary varies across different job characteristics, indicating that job-related attributes contain useful information for salary prediction.

---

## Feature Engineering & Preprocessing

Feature engineering transformed raw job listing information into machine learning inputs.

The workflow included:

1. Target separation
2. Feature selection
3. Removal of irrelevant/non-predictive identifiers
4. Missing-value handling
5. Categorical feature encoding
6. Numerical feature preparation

A structured preprocessing pipeline was used to ensure consistent transformation of training and unseen data.

### Pipeline

Input Features  
↓  
ColumnTransformer  
↓  
Encoded / Transformed Features  
↓  
Regression Model

---

## Machine Learning Methodology

The project follows an end-to-end supervised regression workflow:

**Historical Job Data → Data Cleaning → EDA → Feature Engineering → Preprocessing → Train-Test Split → Model Development → Model Comparison → Hyperparameter Tuning → Final Test Evaluation → Business Interpretation**

The two salary targets were modelled independently:

- Minimum Salary
- Maximum Salary

---

## Models Evaluated

Three regression algorithms were compared:

### 1. Linear Regression

Used as a baseline model to represent a linear relationship between features and salary.

### 2. Random Forest Regressor

Combines multiple decision trees to model complex relationships and improve predictive performance.

### 3. Gradient Boosting Regressor

Builds an ensemble of weak learners sequentially to capture non-linear relationships.

Models were compared using:

- MAE
- RMSE
- R² Score

---

## Model Performance

| Model | Target | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Linear Regression | Minimum Salary | 12,697.67 | 17,833.81 | 0.6041 |
| Linear Regression | Maximum Salary | 14,899.63 | 19,950.81 | 0.7554 |
| Random Forest | Minimum Salary | 3,696.63 | 7,706.09 | 0.9261 |
| Random Forest | Maximum Salary | 6,065.13 | 10,090.05 | 0.9374 |
| Gradient Boosting | Minimum Salary | 11,728.85 | 16,287.89 | 0.6698 |
| Gradient Boosting | Maximum Salary | 14,723.71 | 19,450.84 | 0.7675 |

### Best Performing Model

**Random Forest Regressor** achieved the strongest overall performance for both salary targets.

- Minimum Salary R²: **0.9261**
- Maximum Salary R²: **0.9374**

It also achieved the lowest MAE and RMSE among the evaluated models.

---

## Model Evaluation Metrics

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted salary.

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily.

### R² — Coefficient of Determination

Measures how much of the variation in salary is explained by the model.

Using multiple metrics provides a more complete understanding of prediction accuracy and model reliability.

---

## Hyperparameter Tuning

GridSearchCV was applied to optimize the strongest candidate model.

The tuning process used:

- **24 hyperparameter combinations**
- **5-fold cross-validation**
- **120 total fits**

### Best Random Forest Configuration — Maximum Salary

| Hyperparameter | Selected Value |
|---|---:|
| Number of Estimators | 200 |
| Maximum Depth | None |
| Minimum Samples Split | 2 |
| Minimum Samples Leaf | 1 |

### Best Cross-Validation R²

**0.8832**

The best configuration was selected based on cross-validated R², while the test set remained untouched for final evaluation.

---

## Prediction Analysis

The project also analyzed:

- Actual vs Predicted Salary
- Residual / Error Analysis
- Prediction Error Across Salary Levels
- Feature Importance

Points closer to the diagonal line in actual-vs-predicted plots indicate more accurate predictions.

Residual analysis was used to understand the distribution and magnitude of prediction errors.

Feature importance was analyzed to understand which transformed job attributes contribute most strongly to salary prediction.

**Note:** Feature importance represents model contribution and does not imply a causal relationship.

---

## Business Recommendations

### 1. Recruitment Planning
Use predicted salary ranges as an initial reference when creating job postings.

### 2. Compensation Budgeting
Use salary forecasts to support preliminary workforce budget planning.

### 3. Market Alignment
Compare proposed salary ranges with historical model estimates before publishing job listings.

### 4. Continuous Retraining
Periodically retrain the model using newly available job and salary data.

### 5. Error Monitoring
Monitor prediction errors across job categories and salary levels.

### 6. Human Oversight
Use model predictions as decision support rather than automated salary-setting decisions.

---

## Limitations

- Historical salary patterns may not fully represent current market conditions.
- Some factors influencing salary may not be available in the dataset.
- High-cardinality textual features can be challenging to represent effectively.
- Unusual or rarely represented job postings may produce larger prediction errors.
- External market conditions can change over time.
- Model predictions should not replace human compensation decisions.

---

## Future Scope

The solution can be further improved by:

- Incorporating advanced NLP techniques for job descriptions and preferred skills.
- Adding relevant external salary-market information.
- Testing advanced gradient boosting algorithms.
- Applying SHAP-based explainability.
- Implementing model drift monitoring.
- Developing an interactive recruiter-facing application.
- Automating periodic model retraining.

---

## Conclusion

The Salary Range Prediction project demonstrates an end-to-end machine learning approach for estimating minimum and maximum salaries from historical job listing data.

The workflow included:

**Data Preparation → EDA → Feature Engineering → Model Development → Model Comparison → Hyperparameter Tuning → Evaluation → Business Interpretation**

The results demonstrate that machine learning can provide useful salary estimates and support data-driven recruitment, compensation planning, and workforce decision-making.

### Final Takeaway

**Machine learning can transform historical job data into actionable salary intelligence for better-informed HR decisions.**

---

## Project Files

- `Salary_Range_Prediction.ipynb` — Complete machine learning notebook
- `Salary Range Prediction.pptx` — Project presentation
- `README.md` — Project documentation