import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
   
    if request.method == 'POST':
        Age = float(request.form['Age'])  
        Gender = request.form['Gender']
        Bmi = float(request.form['Bmi'])
        children = int(request.form['children'])
        Smoker = request.form['Smoker']
        region = request.form['region']
        medical_history = request.form['medical_history'] 
        family_medical_history = request.form['family_medical_history']
        exercise_frequency = request.form['exercise_frequency']
        occupation = request.form['occupation']
        coverage_level = request.form['coverage_level']

        new_data = {
            'age': Age,
            'gender': Gender,
            'bmi': Bmi,
            'children': children,
            'smoker': Smoker,
            'region': region,
            'medical_history': medical_history,
            'family_medical_history': family_medical_history,
            'exercise_frequency': exercise_frequency,
            'occupation': occupation,
            'coverage_level': coverage_level
        }
        df = pd.read_csv('insurance_dataset.csv')


        # In[6]:


        print("First few rows of the dataset:")
        df.head()



        X = df.drop('charges', axis=1)
        y = df['charges']
        categorical_features = ['gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level']
        numerical_features = ['age', 'bmi', 'children']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', LinearRegression())])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        model.fit(X_train, y_train)

        # y_pred = model.predict(X_test)
        # Convert to DataFrame
        new_df = pd.DataFrame([new_data])
        predictions = model.predict(new_df)
        
        return render_template('index.html',prediction_text="INSURANCE  amount is {}".format(np.round(predictions,2)))
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

