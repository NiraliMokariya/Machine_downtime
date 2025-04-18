#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
data = pd.read_csv("Machine Downtime.csv")
#print head
print(data.head())
#describe data
print(data.describe())       #gives basic stat things
print(data.info())           #gives datatype etc
print(data.isnull().sum())   #counts null values


#%%
subset = data[['Machine_ID','Assembly_Line_No','Hydraulic_Pressure','Coolant_Pressure','Air_System_Pressure','Coolant_Temperature','Hydraulic_Oil_Temperature','Spindle_Bearing_Temperature','Spindle_Vibration','Tool_Vibration','Spindle_Speed','Voltage','Torque','Cutting','Downtime_Binary']]
sns.pairplot(subset , hue = 'Downtime_Binary')
plt.show()

#%%
subset = data.iloc[:,:14]
subset["Downtime_Binary"]=data["Downtime_Binary"]
sns.pairplot(subset,hue="Downtime_Binary")
plt.show()

#%% drop missing values
data.dropna(inplace=True)
# data.fillna(data.mean(),inplace=True)

#%%
from sklearn.preprocessing import LabelEncoder
data['Assembly_Line_No']=LabelEncoder().fit_transform(data['Assembly_Line_No'])
print(data.isnull().sum())

#%%encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['Machine_ID']=labelencoder.fit_transform(data['Machine_ID'])


# # one hot handling for non binary data
# data = pd.get_dummies(data,columns=['Machine_ID'])
# print(data.head())

#%%
data.head()

#%%
from sklearn.preprocessing import LabelEncoder
data['Assembly_Line_No']=LabelEncoder().fit_transform(data['Assembly_Line_No'])

#%%
# data.describe()
print(data.isnull().sum())
subset=data.iloc[:,:14]
subset['Downtime_Binary']=data["Downtime_Binary"]
sns.pairplot(subset,hue="Downtime_Binary")
plt.show()

#%%
# Select the numerical columns (all columns except 'Downtime_Binary')
numerical_columns = subset.columns[2:-1]  # This excludes the 'Downtime_Binary' column

# Create a boxplot for each numerical column
for col in numerical_columns:
    sns.boxplot(x='Downtime_Binary', y=col, data=subset)
    plt.title(f"Boxplot of {col} grouped by Downtime_Binary")
    plt.show()
    
#%%
data.drop_duplicates(inplace=True)
print(data)
data.info()

#%%
data.drop("Downtime",axis=1,inplace=True)

#%%
plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),annot=True,cmap="Blues")
plt.show()

#%%heatmap
# data_numeric=data.drop(["Downtime"],axis=1)
# plt.figure(figsize=(14,6))
# sns.heatmap(data_numeric.corr(),annot=True,cmap="Blues")
# plt.show()

#%%
sns.kdeplot(data[:14],shade=True)
plt.show()                     #check later

#%% Logistic regression
from sklearn.linear_model import LogisticRegression
x=data.drop('Downtime_Binary',axis=1)
y=data['Downtime_Binary']

#%%  
model = LogisticRegression()
model.fit(x,y)
ypred = model.predict(x)
print(ypred)

from sklearn.metrics import accuracy_score
accuracy_score(ypred,y)

#%%
from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier()
model2.fit(x,y)
ypred2 = model2.predict(x)
accuracy_score(ypred2,y)

#%%
from sklearn.ensemble import RandomForestClassifier

model3 = RandomForestClassifier()
model3.fit(x,y)
ypred3 = model3.predict(x)
accuracy_score(ypred3,y)

#%% classification report
from sklearn.metrics import classification_report
print(classification_report(ypred,y))
print(classification_report(ypred2,y))
print(classification_report(ypred3,y))

#%% confusion matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
ConfusionMatrixDisplay(confusion_matrix(ypred,y)).plot()
ConfusionMatrixDisplay(confusion_matrix(ypred2,y)).plot()
ConfusionMatrixDisplay(confusion_matrix(ypred3,y)).plot()

#%%Split - train and test(logistic regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Prepare your data
X = data.drop('Downtime_Binary', axis=1)  # Features
y = data['Downtime_Binary']  # Target variable

# Step 2: Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Logistic Regression model and set max_iter
model = LogisticRegression(max_iter=300)  # You can adjust max_iter if needed

# Step 4: Train your model on the training set
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

# Step 6: Evaluate your model
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test_Accuracy: {test_accuracy}")
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train_accuracy: {train_accuracy}")

#%% Decision tree
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Prepare your data
X = data.drop('Downtime_Binary', axis=1)  # Features
y = data['Downtime_Binary']  # Target variable

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()

# Step 4: Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#%%Random Forest
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Prepare your data
X = data.drop('Downtime_Binary', axis=1)  # Features
y = data['Downtime_Binary']  # Target variable

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



#%%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model with scaled data
model = LogisticRegression(max_iter=300)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



