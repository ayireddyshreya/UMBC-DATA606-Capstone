# Insurance Dataset for Predicting Health Insurance

## 1. Title and Author

### Project Title
**Insurance Dataset for Predicting Health Insurance**

### Prepared for UMBC Data Science Master Degree Capstone by
**Dr. Chaojie (Jay) Wang**

### Author Name
**Ayireddy Shreya**

### Author's GitHub Profile
https://github.com/ayireddyshreya

### Author's LinkedIn Profile
www.linkedin.com/in/shreya-ayireddy

## Background

The "Insurance Dataset for Predicting Health Insurance" aims to analyze various factors influencing medical costs and health insurance premium. Understanding these factors is crucial for developing accurate predictive models and gaining insights into the relationships among different variables.

### What is it about?

The dataset explores the relationship between individual characteristics (such as age, gender, BMI, smoking status) and external factors (region, income, education, occupation) with health insurance premiums. By studying this dataset, we aim to uncover patterns, correlations, and dependencies that can inform the development of machine learning models for predicting insurance charges.

### Why does it matter?

Understanding the factors influencing health insurance premiums is essential for both insurance providers and individuals seeking coverage. Accurate predictions can lead to better-informed decision-making, personalized pricing, and improved risk assessment in the insurance industry.

### What are your research questions?

1. What is the impact of individual characteristics (age, gender, BMI, smoking status) on health insurance premiums?
2. How do external factors (region, income, education, occupation) contribute to variations in insurance charges?
3. Can a machine learning model accurately predict health insurance premiums based on the provided variables?

## Data

### Data Sources

[Insurance dataset](https://www.kaggle.com/datasets/sridharstreaks/insurance-data-for-machine-learning/data)

### Data Size

The dataset size is substantial, comprising a million records. The file size is 101.94 MB.

### Data Shape

The dataset consists of 10 variables/columns and 1 million rows.

### Time Period

The data is not time-bound as it was synthetically generated for diverse representation.

### Each Row Represents

Each row represents an insured individual and includes information on their age, gender, BMI, number of children, smoking status, region, income, education, occupation, and type of insurance plan.

## Data Dictionary

### Data Dictionary

| Column Name           | Data Type | Definition                                      | Potential Values                                      |
|------------------------|-----------|-------------------------------------------------|--------------------------------------------------------|
| Age                    | Integer   | The age of the insured individual.              | Integer values                                         |
| Gender                 | Object    | The gender of the insured individual.           | 'Male' or 'Female'                                     |
| BMI (Body Mass Index) | Float     | A measure of body fat based on height and weight.| Float values                                           |
| Children               | Integer   | The number of children covered by the insurance plan.| Integer values                                     |
| Smoking Status        | Object    | Indicates whether the individual is a smoker.  | 'Smoker' or 'Non-Smoker'                               |
| Region                 | Object    | The geographical region of the insured individual.| 'North', 'South', 'East', 'West'                 |
| Medical History       | Object    | Information about the individual's historical medical problems.| Categorical values                       |
| Family Medical History | Object    | Information about the medical history of the family.| Categorical values                               |
| Exercise Frequency     | Object    | The frequency of the individual's exercise routine.| 'Never', 'Rarely', 'Occasionally', 'Frequently'                              |
| Occupation             | Object    | The occupation of the insured individual.       | Categorical values                                     |
| Coverage Level         | Object    | The type of insurance plan.                     | 'Basic', 'Standard', 'Premium'                         |
| Charges               | Float     | The health insurance charges for the individual.| Float values                                           |


### Target/Label and Features for ML Models

- **Target/Label:** 
In the provided information, the target/label variable for the machine learning model is the 'Charges' column. This variable represents the health insurance charges for the individual and would be the predicted value in the machine learning model.
- **Features/Predictors:** 
'gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', and 'coverage_level'  can be used as variables for predicting machine learning models. These variables represent categorical features, and depending on the nature of your predictive task, they can be valuable predictors for your model.
