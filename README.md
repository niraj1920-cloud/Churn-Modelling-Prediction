
# Churn Modelling Prediction using ANN



## About the Project
Developed a predictive model to identify potential customer churn, enabling proactive customer retention strategies. The model leveraged an Artificial Neural Network (ANN) to classify customers based on their likelihood to churn, helping businesses minimize revenue loss.

## About Dataset
This dataset contains information about customers of a bank, with various attributes related to their demographics, financial habits, and relationship with the bank. It is commonly used for customer churn analysis, where the goal is to predict whether a customer is likely to "churn" (leave the bank).

Dataset Features:
Independent variables

- RowNumber: Index of the row in the dataset.
- CustomerId: Unique identifier for each customer.
- Surname: Last name of the customer (not relevant for prediction, can be removed).
- CreditScore: Credit score of the customer, indicating their creditworthiness.
- Geography: Country where the customer resides.
- Gender: Gender of the customer.
- Age: Age of the customer.
- Tenure: Number of years the customer has been with the bank.
- Balance: Account balance of the customer.
- NumOfProducts: Number of products the customer has with the bank.
- HasCrCard: Indicates whether the customer has a credit card (1 = Yes, 0 = No).
- IsActiveMember: Indicates if the customer is an active member (1 = Active, 0 = Inactive).
- EstimatedSalary: Estimated annual salary of the customer.
Target variable:

- Exited: Target variable, where 1 indicates the customer has left the bank, and 0 means they have not.
## Tech Stack
- Python
- Pandas
- Numpy
- Tensorflow
- Keras
- Streamlit
## Project Structure

```bash
  Churn-Modelling-Prediction/
│
├── Churn_Modelling.csv              # Dataset file
├── app.py                           # Main application script for running the model
├── experiments.ipynb                # Initial exploratory data analysis and experimentation
├── hyperparametertuningann.ipynb    # Hyperparameter tuning for ANN model
├── prediction.ipynb                 # Notebook for prediction evaluation
├── salaryregression.ipynb           # Regression model for salary prediction
├── model.h5                         # Trained ANN model for churn prediction
├── regression_model.h5              # Trained regression model for salary prediction
├── label_encoder_gender.pkl         # Saved label encoder for 'Gender' feature
├── one_hot_encoder_geo.pkl          # Saved one-hot encoder for 'Geography' feature
├── scaler.pkl                       # Saved scaler for feature scaling
├── requirements.txt                 # List of dependencies for the project (e.g., TensorFlow, Pandas, etc.)
│
└── logs/
    ├── fit20240815-161446/          # Folder for model training logs
    │   ├── train/                   # Logs from training phase
    │   └── validation/              # Logs from validation phase
    │
    └── regressionlogs/              # Folder for regression model training logs

```
## Workflow
1.  Data Collection and Loading:
Load the dataset Churn_Modelling.csv, which contains information about bank customers, such as demographics, credit score, account balance, and other factors relevant to churn prediction.

2. Data Preprocessing:
- Encoding categorical features:
  
  - Use LabelEncoder to convert the Gender column into numerical values (e.g., Male to 0, Female to 1), which simplifies processing by the model.

    ```bash
    ##Encode categorical variables
    label_encoder_gender=LabelEncoder()
    data['Gender']=label_encoder_gender.fit_transform(data  ['Gender'])

     ```
  - This encoding is saved as label_encoder_gender.pkl so it can be reused consistently during inference.
  - Use OneHotEncoder to transform the Geography column, creating separate binary columns for each unique geography (e.g., France, Spain, Germany).

    ```bash
    ##Onehot encode 'Geography'
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder_geo=OneHotEncoder()
    geo_encoder=onehot_encoder_geo.fit_transform(data[['Geography']])

     ```

  - After encoding, the new columns are concatenated with the original DataFrame.
  - Save this one-hot encoder as one_hot_encoder_geo.pkl for consistency in future predictions.

 -  Splitting the Dataset
    - Train-Test Split: Split the data into training and test sets to evaluate the model's performance. This is necessary to ensure the model generalizes well on unseen data.

        ```bash
        ##Divide the dataset into dependent and independent     features
        x=data.drop('Exited',axis=1)
        y=data['Exited']

        ##Split the data in training and testing sets
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

        ```

  - Feature Scaling:
    - Use StandardScaler to scale the features, which normalizes numerical data by removing the mean and scaling to unit variance.
    - Fit the scaler on the training data (X_train) and then transform both X_train and X_test for consistency.
    - Save the scaler as scaler.pkl for use in real-time predictions.

3. Building the ANN:
- Initialize a Sequential model using TensorFlow and Keras.
- Define the layers:
  - Input Layer: The input shape is specified as (X_train.shape[1],) to match the number of features in the dataset.
  - Hidden Layer 1: Add the first hidden layer with 64 neurons and ReLU activation. This layer is fully connected to the input layer, helping the model learn complex patterns in the data.
  - Hidden Layer 2: Add a second hidden layer with 32 neurons and ReLU activation for additional feature extraction.
  - Output Layer: Add a single neuron with sigmoid activation. The sigmoid function is ideal for binary classification problems, as it outputs probabilities between 0 and 1 (indicating the likelihood of churn).

    ```bash
    model=Sequential([
        Dense(64,activation='relu',input_shape=(x_train.shape[1],)), ##HL1 connected with input layer,64 neurons
        Dense(32,activation='relu'), ##HL2
        Dense(1,activation='sigmoid') ##output layer
    ])
    ```

4. Making Predictions and Evaluating the Model
  - Predictions:
    Use the trained model to make predictions on the test set to assess its performance on unseen data.
  - Evaluation:The accuracy metric is used to track the model’s performance by calculating the percentage of correct predictions.
    

5. Saving and Deployment
 - Save Model: Save the trained ANN model as model.h5 for future use and deployment.
- Deploy: Implement app.py to create a user interface using Streamlit web framework.

## Screenshots
## Contributions
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request
