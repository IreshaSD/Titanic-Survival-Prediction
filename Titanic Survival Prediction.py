import numpy as np  # Library for creating NumPy arrays
import pandas as pd  # Library for working with structured tables (dataframes)
import matplotlib.pyplot as plt  # Library for data visualization - plots and graphs
import seaborn as sns  # Library for data visualization - plots and graphs
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.metrics import accuracy_score  # Evaluation metric for model assessment

# Load the data from a .csv file into a Pandas DataFrame
titanic_data = pd.read_csv("/content/Titanic_Survival.csv")

# Display the first five rows of the dataframe
titanic_data.head()

# Check the number of rows and columns
titanic_data.shape

# Get information about the data
titanic_data.info()

# Check the number of missing values in each column
titanic_data.isnull().sum()

# Drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns="Cabin", axis=1)  # Dropping a column using axis=1

# Replace missing values in "Age" column with mean value
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)  # Fill missing values with mean

# Replace missing values in "Embarked" column with mode value
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)  # Fill with mode

# Check if missing values are replaced correctly
titanic_data.isnull().sum()

# Get statistical measures about the data
titanic_data.describe()

# Find the number of people survived and not survived
titanic_data["Survived"].value_counts()

# Visualize counts of 'Survived' column using a count plot
sns.countplot(x="Survived", data=titanic_data)

# Visualize counts of 'Sex' column using a count plot
sns.countplot(x="Sex", data=titanic_data)

# Compare survival counts based on gender
sns.countplot(x="Sex", hue="Survived", data=titanic_data)

# Visualize counts of 'Pclass' column using a count plot
sns.countplot(x="Pclass", data=titanic_data)

# Compare survival counts based on Passenger's class
sns.countplot(x="Pclass", hue="Survived", data=titanic_data)

# Visualize counts of 'Embarked' column using a count plot
sns.countplot(x="Embarked", data=titanic_data)

# Compare survival counts based on Embarked port
sns.countplot(x="Embarked", hue="Survived", data=titanic_data)

# Visualize counts of 'Age' column using a count plot
sns.countplot(x="Age", data=titanic_data)

# Compare survival counts based on Age
sns.countplot(x="Age", hue="Survived", data=titanic_data)

# Replace categorical columns with numerical values
titanic_data1 = titanic_data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}})

x = titanic_data1.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
y = titanic_data1["Survived"]

print(x)
print(y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

model = LogisticRegression()

# Train the Logistic Regression model with training data
model.fit(x_train, y_train)

# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

print("Accuracy score of the training data:", training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

print("Accuracy score of the test data:", test_data_accuracy)
