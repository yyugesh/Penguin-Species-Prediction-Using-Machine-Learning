
# Penguin-Species-Prediction Using Machine Learning

#### This project aims to classify the species of penguins using machine learning techniques based on features like species, island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex. It uses the penguins_size dataset for the classic supervised classification tasks.


## Tech

 **Language**                `Python`,

 **Data Handling**           `pandas`, `numpy`

 **Data Visualization**      `matplotlib`, `seaborn`

 **Machine Learning**        `scikit-learn`

 **Jupyter Environment**     `Jupyter Notebook` or `Google Colab`

 **code editors/IDEs**       `Pycham` or `VS Code (Visual Studio Code)`

 **Version Control**         `git`, `GitHub`
 
 **Environment Management**  `venv`

## Environment Variables

1. What environment variables are needed

2. Why they’re used

### Prerequisites

Make sure you have the following installed:

1. Python 3.7 or higher

2. pip (Python package manager)

3. Git (for cloning the repository)


### Step 1: Clone the Repository

#### Open your terminal or command prompt and run:

git clone https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning.git

### Step 2: Set Up a Virtual Environment

#### Windows
python -m venv venv
venv\Scripts\activate

#### macOS/Linux
python3 -m venv venv
source venv/bin/activate



## Installation & Running the Project

Follow these steps to install and run the Penguin Species Prediction using Machine Learning project on your machine.

### Step 1: Install Required Libraries
    
Run the following commands to install the necessary dependencies:

1. Install **pandas**

    ```
    pip install pandas
    ```

2. Install **numpy**

    ```
    pip install numpy
    ```

3. Install **matplotlib**

    ```
    pip install matplotlib
    ```

4. Install **seaborn**

    ```
    pip install seaborn
    ```

5. Install **scikit-learn**

    ```
    pip install scikit-learn
    ```

Or if you have a requirements.txt file:

    ```
    pip install -r requirements.txt
    ```

### Step 2: Run the project 

Jupyter Notebook, VS Code, or PyCharm, among others are the popular code editors/IDEs.

## Features

- **Penguin Species Classification**: Predicts species (*Adélie*, *Gentoo*, *Chinstrap*) based on physical characteristics.

- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numeric data.

- **Exploratory Data Analysis (EDA)**: Visualizes relationships using pair plots, box plots, and heatmaps.

- **Machine Learning Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)

- **Model Evaluation**: Assesses performance using metrics like Accuracy, Confusion Matrix, and Classification Report.

- **Train-Test Split**: Ensures unbiased model evaluation.

- **Organized Codebase**: Clean structure for data loading, preprocessing, training, and testing.

- **Ready for Deployment**: Easily adaptable to tools like Streamlit or Flask for web app integration.
## Screenshots

### 1. Pre-processing Data Analysis
![PDA Screenshot](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_1.png)

### 2. Exploratory Data Analysis
![EDA Screenshot](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_2.png)
![EDA Screenshot](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_3.png)

### 3. Model Prediction Output
![Prediction Result 1](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_4.png)
![Prediction Result 2](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_5.png)

![Prediction Result 3](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_6.png)

![Prediction Result 4](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_7.png)

![Prediction Result 5](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_8.png)

### 4. Model Accuracy Comparison Output
![Model Accuracy](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/55f3b4084d0748dd2e5bc24e562878c85a2013b9/Figure_9.png)

## Demo

Check out the demo of the Penguin Species Prediction app:

[Click here to try the app](https://github.com/yyugesh/Penguin-Species-Prediction-Using-Machine-Learning/blob/main/Penguin-Species-Prediction%20Using%20Machine%20Learning.gif)  

## About the Dataset

- The dataset used in this project is `penguins_size.csv`, originally provided by Dr. Kristen Gorman

- It includes data collected from three penguin species (*Adélie*, *Chinstrap* and *Gentoo*).

- Key features include:
  - **Island** (location of observation)
  - **culmen_length_mm**
  - **culmen_depth_mm**
  - **flipper_length_mm**
  - **body_mass_g**
  - **Sex**

- Missing values were handled through imputation or row removal to ensure model reliability.

- The dataset is popular for demonstrating classification tasks in data science and machine learning education.

- The penguins_size.csv dataset contains 344 observations of penguins across 7 features, used to classify three penguin species.


### Notes
Some rows contain missing values, especially in sex and numerical features.

culmen_length_mm and culmen_depth_mm are commonly referred to as bill length and bill depth in documentation.

The dataset is well-suited for demonstrating data cleaning, EDA, and classification using supervised learning algorithms.
## Authors

- [@ Dr. Kristen Gorman](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data?select=penguins_size.csv)

