# Step 1: Define the problem

# The problem is to classify the type of Iris plant (Setosa, Versicolor, or Virginica) based on sepal length, sepal width, petal length, and petal width

# Step 2: Collect and clean the data

# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset

# before loading the dataset as a dataframe, let explore how it originally exist as a bunch type. The bunch type
#create different objects with the dataset

#the most important objects are the numerical data (X values or independent variables) and the numerical target (y values or dependent variables)
iris_data = load_iris()
print(iris_data.data)
print(iris_data.feature_names)
print(iris_data.target)
print(iris_data.target_names)

# Create a dataframe from the Iris data
df = pd.DataFrame(iris_data['data'], columns=iris_data['feature_names'])
#df['target'] = iris_data['target']
#df['target_name'] = iris_data['target_names']


# Step 3: Explore and visualize the data

# Import necessary libraries
import matplotlib.pyplot as plt

# Explore the data
#print(df.head())
#print(df.describe())

# Visualize the data

#as a scatter plot
#df.plot(x='sepal length (cm)', y='sepal width (cm)', kind='scatter')
#plt.show()

#as a histogram
#df.hist()
#plt.show()

# Step 4: Preprocess the data and split into train and test

# Split the data into train and test sets
#I am getting an error defining X
#maybe i dont need to get the X and y values from the dataframe df?

from sklearn.model_selection import train_test_split

X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train and evaluate models

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Step 6: Communicate the results

# Report the findings and accuracy of the model
print("The logistic regression model accurately predicted the type of Iris plant with an accuracy of", accuracy)
