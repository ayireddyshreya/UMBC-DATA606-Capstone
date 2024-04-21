# Predicting Health Insurance using ML techniques


## 1. Title and Author

- **Project Title:** Predicting health insurance using ML techniques
- **Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang**
- **Author Name:** Shreya Ayireddy
- **GitHub Profile:**
https://github.com/ayireddyshreya
- **LinkedIn Profile:**
www.linkedin.com/in/shreya-ayireddy
- **Presentation file:**
https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/Presentation.pptx
- **Youtube:**


## 2. Background

The "Predicting Health Insurance" aims to analyze various factors influencing medical costs and health insurance premium. Understanding these factors is crucial for developing accurate predictive models and gaining insights into the relationships among different variables.

### What is it about?

The dataset explores the relationship between individual characteristics (such as age, gender, BMI, smoking status) and external factors (region, occupation) with health insurance premiums. By studying this dataset, I aim to uncover patterns, correlations, and dependencies that can inform the development of machine learning models for predicting insurance charges.

### Why does it matter?

Understanding the factors influencing health insurance premiums is essential for both insurance providers and individuals seeking coverage. Accurate predictions can lead to better-informed decision-making, personalized pricing, and improved risk assessment in the insurance industry.

### What are your research questions?

1. What is the impact of individual characteristics (age, gender, BMI, smoking status) on health insurance premiums?
2. How do external factors (region, income, education, occupation) contribute to variations in insurance charges?
3. Can a machine learning model accurately predict health insurance premiums based on the provided variables?

## 3. Data

### Data Sources

[Insurance dataset](https://www.kaggle.com/datasets/sridharstreaks/insurance-data-for-machine-learning/data)

### Data Size

The dataset size is substantial, comprising a million records. The file size is 101.94 MB.

### Data Shape

The dataset consists of 12 variables/columns and 1 million rows.

### Time Period

The data is not time-bound as it was synthetically generated for diverse representation.

### Each Row Represents

Each row represents an insured individual and includes information on their age, gender, BMI, number of children, smoking status, region, income, education, occupation, and type of insurance plan.

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



### Target/Label
In the provided information, the target/label variable for the machine learning model is the 'Charges' column. This variable represents the health insurance charges for the individual and would be the predicted value in the machine learning model.

### Features/Predictors
'Gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', and 'coverage_level'  can be used as variables for predicting machine learning models. These variables represent categorical features, and depending on the nature of your predictive task, they can be valuable predictors for your model.

## 4. Exploratory Data Analysis
There are no missing values and duplicate rows in the dataset. 

### Summary Statistics:

**Key Variables:**

- **Charges Distribution:**
  - Minimum Charges: 3445.01
  - Maximum Charges: 32561.56

- **Age Distribution:**
  - Minimum Age: 18
  - Maximum Age: 65

- **BMI Distribution**
  - Minimum BMI: 18
  - Maximum BMI: 50

### Distribution of Health Insurance Charges
- **Minimum Charges:** 3445.0116431134834
- **Maximum Charges:** 32561.56037356053

### Visualizations

**Distribution of the target variable 'Charges'**
![1](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/1.png)
The bars on the graph depict the number of people who have incurred a certain range of charges. For instance, there appear to be more people with charges between 5000 and 10,000 than those with charges between 25,000 and 30,000.

**Distribution of Individuals across Regions**
![2](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/2.png)
We can see that the dataset contains information of people from northeast more than other three regions. However, there is no big difference in the count of individuals but it's slightly varying.

**BMI distribution by smoking status**
![3](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/3.png)
In the dataset, BMI starts from 18 and ranges till 50. There is no data for the person before 18. 

In this graph, it shows the relative number of people who have a certain BMI. 

The higher the density at a particular BMI value, the more people there are with that BMI. The density is highest for non-smokers at a BMI of around 28-29. This means that there are more people who do not smoke with a BMI around 28-29.

**Distribution of Charges by Smoker Status**
![4](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/4.png)
In this graph, the box for “no” smokers is slightly lower than the box for “yes” smokers, which means that non-smokers have a lower median charge than smokers.

**Relationship between age and charges**
![5](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/5.png)
Here in the dataset, age starts from 18 and ranges till 65. There is no data for the person before 18 years. So, x-axis in the graph starts from 18 and continues till 65 with the interval of 10.

Charges and Age: There seems to be a positive correlation between age and charges for both smokers and non-smokers. This means that as age increases, the number of people with charges also increases.

Smoking and Charges: The graph suggests that there might be fewer non-smokers with charges compared to smokers across all age groups. The green lines (non-smokers) generally appear lower than the blue lines (smokers) throughout the graph.

**Count of Smokers and Non-Smokers**
![6](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/6.png)
The blue line represents the number of smokers, while the orange line represents the number of non-smokers. Smokers are more compared to non-smokers. Number of smokers are 500,129. Whereas number of non-smokers are 499,871.

**Distribution of charges by region**
![7](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/7.png)
The box for the southwest region appears to have a lower median and a smaller IQR(interquartile range) than the box for the northeast region. This suggests that the charges in the southwest region tend to be lower and less spread out than the charges in the northeast region. There are also appears to be more outliers in the northeast region than the southwest region. The northwest and southeast regions have similar medians, but the northwest region has a slightly larger IQR, indicating slightly more spread-out charges.

**Relationship between BMI and charges colored by smoker status**
![8](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/8.png)
Here in the dataset, BMI starts from 18. There is no data for the person before 18. So, x-axis in the graph starts from 18 and continues with the interval of 5 till 50 which is the max value of BMI in the dataset.

The y-axis shows the total charge amount incurred by people with a certain Body Mass Index (BMI) for each smoking status. In essence, it depicts the relationship between BMI and healthcare charges, while accounting for whether the person smokes or not.

The graph suggests that there is a positive correlation between BMI and charges for both smokers and non-smokers. This means that as BMI increases, the total charges tend to increase as well. However, the increase seems to be steeper for non-smokers (blue line) compared to smokers (green line). This suggests that non-smokers with a higher BMI tend to incur more charges than smokers with a higher BMI.

**Insurance charges distribution by coverage level**
![9](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/9.png)
Charge Distribution: The most frequent charges (represented by the tallest bar) fall between 5,000 and 10,000 across all coverage levels. Coverage Level: It appears there are more people with standard coverage than those with basic or premium coverage (based on the relative heights of the bars).

**Correlation Matrix**
![10](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/10.png)
Each cell in the matrix contains a correlation coefficient, which is a number between -1 and 1 that indicates the strength and direction of the relationship between two variables.

Warmer color and a correlation coefficient of 1 indicates a perfect positive correlation, meaning that as the value of one variable increases, the value of the other variable also increases. Cooler color and a correlation coefficient of -1 indicates a perfect negative correlation, meaning that as the value of one variable increases, the value of the other variable decreases.

## 5. Model Training
By leveraging machine learning algorithms, we can build predictive models that estimate health insurance charges based on factors such as age, BMI, smoking status, and region.

Regression models used in this project are:

1. Decision Tree Regression
2. Random Forest Regression
3. Linear Regression

***Data preprocessing***

Data preprocessing is a critical step in machine learning workflows to ensure data compatibility with models.
Standardization and encoding techniques are applied to handle numerical and categorical features, respectively.
This step ensures that the data is in a suitable format for model training.

***Model building***

The use of pipelines simplifies the workflow by combining preprocessing and modeling steps.
Utilization of the Pipeline class from Scikit-learn to combine preprocessing and modeling steps.
ColumnTransformer allows for applying different transformations to different types of features.
Utilization of the Pipeline class from Scikit-learn to combine preprocessing and modeling steps.

***Model evaluation***

Splitting the dataset into training and testing sets using train_test_split.
Fitting the model on the training data using the fit method.
Making predictions on the testing data using the predict method.
Calculation of evaluation metrics: 
- Mean Absolute Error (MAE) 
- Mean Squared Error (MSE)
- R-squared

**Results**
- ***Linear regression***
  - Mean Absolute Error: 250.34
  - Mean Squared Error: 83483.88 
  - R-squared: 0.9957 
- ***Desicion Tree regressor***
  - Mean Squared Error: 329418.30 
  - Mean Absolute Error:  457.84 
  - R-squared:  0.9831 

- ***Random Forest regressor***
  - Mean Squared Error: 170077.18 
  - Mean Absolute Error: 334.78 
  - R-squared: 0.9913

![11](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/11.png)


Linear Regression model emerges as the preferred choice for predicting healthcare charges due to its outstanding performance metrics
Linear Regression exhibits the lowest MSE and MAE, indicating superior predictive accuracy and precision.
With the highest R-squared value among the models, Linear Regression explains the most variance in the data.
The exceptional performance of Linear Regression makes it the recommended choice for predicting healthcare charges.

## Application of the Trained Models
Flask web application utilizes a pre-trained machine learning model to predict insurance charges based on user input. 
Users input their demographic and health-related information, including age, gender, BMI, smoking status, and region, through a simple web form. This data is then transmitted to a pre-trained machine learning model, which utilizes historical insurance data to make predictions. 
The model predicts insurance charges based on the provided information, returning the estimated costs to the user interface for display. By integrating machine learning with web technology, this application exemplifies a practical use case, enhancing accessibility to insurance pricing information for a wider audience.

![12](https://github.com/ayireddyshreya/UMBC-DATA606-Capstone/blob/main/docs/12.png)

## Conclusion
My analysis involved robust data preprocessing and model building, utilizing Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
The developed models demonstrated strong predictive performance in estimating healthcare charges, with the Linear Regression model exhibiting exceptional accuracy.
- Implications: The accurate prediction of healthcare charges can facilitate informed decision-making and resource allocation in healthcare management, aiding in budgeting and financial planning.
The model's accuracy enables informed decision-making and resource allocation in healthcare management.
Applications extend to healthcare cost prediction, budgeting, and identifying cost drivers.
- Limitations:
  - Model Assumptions: Linear Regression models rely on assumptions such as linearity, independence of errors, which may not always hold true in real-world healthcare data. Violations of these assumptions can lead to biased estimates and inaccurate predictions.
  - Ethical and Legal Considerations: The use of predictive models in healthcare raises ethical and legal concerns related to patient privacy, fairness, and potential biases in decision-making. Ensuring transparency, accountability, and adherence to regulatory requirements is essential to mitigate these risks.
- Future Scope:
   - Integration of Advanced Machine Learning Techniques: Explore the integration of advanced machine learning techniques such as deep learning and neural networks to capture intricate patterns and dependencies in healthcare data.
   - Integration with Value-Based Care Initiatives: Integrate predictive analytics models with value-based care initiatives to promote cost-effective and high-quality healthcare delivery. 

## References
- Badawy, M., Ramadan, N. & Hefny, H.A. Healthcare predictive analytics using machine learning and deep learning techniques: a survey. Journal of Electrical Systems and Inf Technol 10, 40 (2023). https://doi.org/10.1186/s43067-023-00108-y

- Krishnamoorthi R, Joshi S, Almarzouki H Z, Shukla P K, Rizwan A, Kalpana C, & Tiwari B (2022) A novel diabetes healthcare disease prediction framework using machine learning techniques. J Healthcare Eng. https://doi.org/10.1155/2022/1684017

- Goyal P, Pandey S, Jain K, Goyal P, Pandey S, Jain K (2018) Introduction to natural language processing and deep learning. Deep Learn Nat Language Process: Creat Neural Netw Python 1–74. https://doi.org/10.1007/978-1-4842-3685-7
