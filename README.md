# Customer Churn Prediction using Artificial Neural Network (ANN)
This project is focused on predicting customer churn in a telecom company using an Artificial Neural Network (ANN). The aim is to help telecom businesses identify customers who are likely to stop using their services, so they can take action to retain them. The dataset used includes customer demographics, account information, and service usage patterns.
The neural network model was built using Keras with TensorFlow backend and trained on cleaned, preprocessed data. The project also includes detailed analysis and performance evaluation using metrics like accuracy, precision, recall, and F1-score.
## Problem Statement
Customer churn is a major issue in the telecom industry. Losing customers means losing recurring revenue. By predicting which customers are at risk of leaving, companies can take proactive steps to keep them. This project uses a deep learning approach to solve this classification problem.
## Dataset Details
The dataset includes the following features:

(i) Demographic Info: Gender, Senior Citizen, Partner, Dependents

(ii) Service Details: Internet Service, Online Security, Tech Support, Streaming Services

(iii) Account Info: Tenure, Monthly Charges, Total Charges, Contract Type, Payment Method

(iv) Target Variable: Churn (Yes/No)

(v) The original dataset had over 7000 entries and required some cleaning before it could be used for training.

## Steps Followed:
### 1. Data Cleaning and Preprocessing
(i) First I checked the dataset using df.head() and df.info() to understand the structure.

(ii) There were a few rows where TotalCharges was blank. I removed those.

(iii) I also changed the SeniorCitizen column to categorical (0 or 1) and converted other categorical columns into numbers using Label Encoding.

(iv) Then I applied One Hot Encoding for some categorical features that had more than 2 values.

(v) After encoding, I normalized the values using StandardScaler to bring all values into a similar range.

### 2. Splitting the Data
(i) I separated the target column Churn from the features.

(ii) Then I used train_test_split to divide the dataset into training and testing sets with 80% for training and 20% for testing.

### 3. Building the ANN Model
(i) I used the Sequential API from Keras.

(ii) The model has:

Input layer based on the number of features after preprocessing

Two hidden layers with ReLU activation

One output layer with sigmoid activation (since it’s binary classification)

(iii) I compiled the model using:

Optimizer: adam

Loss function: binary_crossentropy

Metrics: accuracy

(iv) Then I trained the model using model.fit() for 100 epochs with a batch size of 32.

### 4. Model Evaluation
(i) After training, I predicted the output for the test set using model.predict(X_test) and converted the probabilities to 0 or 1.

(ii) Then I used the following metrics to evaluate the model:

Accuracy Score

Confusion Matrix

Precision

Recall

F1 Score

### 5. Results
(i) The model gave a good accuracy on the test data.

(ii) Confusion Matrix showed that the model was correctly identifying both churned and non-churned customers.

(iii) The precision and recall were also balanced, which means the model didn’t just focus on one class.

## Tools & Technologies Used
(i) Python

(ii) Pandas, NumPy for data handling.

(iii) Matplotlib, Seaborn for visualization and EDA.

(iv) Scikit-learn for preprocessing and evaluation metrics.

(v) Keras and TensorFlow for building and training the ANN.

## What I Learned
(i) I learned how to handle categorical data using both label and one-hot encoding.

(ii) Understood how to normalize data before feeding it into a neural network.

(iii) Learned how to build an ANN using Keras Sequential API.

(iv) Understood how to use evaluation metrics like confusion matrix, precision, recall, and F1 score to judge the performance of a model.

(v) The importance of feature scaling and data preprocessing in deep learning projects.






