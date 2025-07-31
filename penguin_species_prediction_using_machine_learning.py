# Imports Necessary Libraries
import pandas as pd
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# Data Collection (Loading Data)
'''
Data Collection is the process of gathering and acquiring data from various sources 
to be used for analysis, modeling, or research. 
'''
df = pd.read_csv("penguins_size.csv")      # Load the dataset

# Data Inspection / Exploratory Data Analysis (EDA)

print(df.head(), df.info(), df.isnull().sum(), df.describe())
print("*" * 42)

# Data Manipulation --> Preprocessing
'''
Data Manipulation is the process of cleaning, transforming, and restructuring data 
to prepare it for analysis or modeling.
'''

# Drop Missing
df.dropna(inplace=True)                      # Drop rows with missing values / Removing null values

# Count plot
plt.figure(figsize=(12, 6))
sns.countplot(x='species', data=df, palette='YlGnBu')
plt.title("Count of Each Penguin Species")
plt.show()

# Value counts
print(df['species'].value_counts())

# Encode Categorical Features with Normalization or Scaling
label_encoders = {}                          # Initializes an empty dictionary
for col in ['species', 'island', 'sex']:     # loop iterates the list of categorical columns in the dataset that contain string labels.
    le = LabelEncoder()                      # Creates a new instance of LabelEncoder from sklearn.preprocessing.
    df[col] = le.fit_transform(df[col])      # Fits the LE to the column "col" and transforms the string values into numeric.
    label_encoders[col] = le                 # Saves and trained LE in the label_encoders dictionary.

print(df.head())

# Grouping and aggregating (groupby()),# Merging or joining datasets, # Creating new columns or features

# Data Inspection / Exploratory Data Analysis (EDA)

#  Univariate Analysis
# Histogram
plt.figure(figsize=(12,6))
sns.histplot(df['flipper_length_mm'], kde=True)  # Numerical Column --> flipper_length_mm
plt.title("Distribution of Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Frequency")
plt.show()

#  Bivariate Analysis
# Drop categorical columns
num_df = df.select_dtypes(include='number')  #  Correlation Heatmap (for numerical columns)

# Correlation matrix
corr = num_df.corr()

# Heatmap
plt.figure(figsize=(12,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# Features and Target
X = df.drop("species", axis=1)  # All except target / Independent variables [features / inputs]
y = df["species"]                     # Dependent variable [Target / Output / Label]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train            [Feature data used to train the model]
# X_test             [Feature data used to test the model (after training)]
# y_train            [Target values for the training dataset]
# y_test             [Target values for the test dataset]
# X                  [Features (independent variables)]
# y                  [Target variable (dependent variable)]
# test_size=0.2	     [20% of the data goes to the test, and 80% go for training set]
# random_state=42	 [Controls the random split. Using the same number gives you reproducible results every time.]

# Models Dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "KNN": KNeighborsClassifier()
}

'''
 Created a models dictionary with key name ML models strings, and 
 values are initialized through model objects from scikit-learn.
'''

# Training and Evaluation
results = {}                                     # Dictionary to store accuracy scores

for name, model in models.items():               # Loop through each model
    model.fit(X_train, y_train)                  # Train the model
    y_pred = model.predict(X_test)               # Predict on test data

    results[name] = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Visualizing the Confusion Matrix using Seaborn
    cm = confusion_matrix(y_test, y_pred)                        # Plot confusion matrix using seaborn
    plt.figure(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu",
                xticklabels=label_encoders['species'].classes_,  # xticklabels : labels from original encoded species column
                yticklabels=label_encoders['species'].classes_)  # yticklabels : labels from original encoded species column
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Visualizing Summary
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="Spectral")
'''
x=list(results.keys())   : uses model names (e.g., "Logistic Regression", "SVC") on the x-axis.
y=list(results.values()) : uses corresponding accuracy scores on the y-axis.
palette="Spectral"       : applies a visually appealing color gradient from the Viridis colormap.
'''
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()