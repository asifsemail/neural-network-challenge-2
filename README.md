# Neural Network Challenge - Module19: Employee Attrition and Department Prediction

## Author
- **Name:** Asif Khan  
- **Date:** January 2025  
- **Module:** Module 19 Challenge  
- **Project:** Module19 - Neural Network Challenge: Employee Attrition and Department Prediction  

---

## **Overview**

This project involves creating a branched neural network to assist HR in two tasks:
1. Predicting whether employees are likely to leave the company (Attrition).
2. Recommending the most suitable department for employees (Department).  

The model uses features from employee data, applies preprocessing, and trains a neural network with two output branches: one for predicting attrition and the other for department classification.

---

## **Objectives**

### **Part 1: Preprocessing**
1. **Data Import and Inspection**: Import employee data and explore its structure.
2. **Identify Targets and Features**: Define `Attrition` and `Department` as target variables and select at least 10 features for prediction.
3. **Preprocess Data**:
   - Encode categorical variables using OneHotEncoding.
   - Scale numeric variables with StandardScaler.
4. **Split Data**: Split the dataset into training and testing sets for model training and evaluation.

### **Part 2: Create, Compile, and Train the Model**
1. **Model Design**:
   - Create input layers and shared layers for the neural network.
   - Design two branches for the outputs: one for `Attrition` and one for `Department`.
2. **Compile and Train the Model**:
   - Compile the model with appropriate loss functions (`categorical_crossentropy` for Department, `binary_crossentropy` for Attrition).
   - Train the model with 100 epochs and a batch size of 32.
3. **Evaluate the Model**: Assess model performance using metrics like accuracy and loss for both branches.

### **Part 3: Summary**
1. **Evaluate Metrics**: Determine if accuracy is the best metric for this task.
2. **Activation Functions**: Justify the activation functions chosen for each output branch.
3. **Model Improvement**: Suggest ways to improve model performance.

---

## **Dataset**

### **Source**
[Dataset: attrition.csv](https://static.bc-edx.com/ai/ail-v-1-0/m19/lms/datasets/attrition.csv)
The dataset contains employee information, including demographics, work experience, and other features.

### **Key Features**
- **Attrition** (Target 1): Binary classification (Yes/No) to predict employee likelihood of leaving the company.
- **Department** (Target 2): Multi-class classification (Sales, Research & Development, Human Resources) to recommend the best-fit department.

### **Features Selected for X**
1. Education  
2. Age  
3. DistanceFromHome  
4. JobSatisfaction  
5. OverTime  
6. StockOptionLevel  
7. WorkLifeBalance  
8. YearsAtCompany  
9. YearsSinceLastPromotion  
10. NumCompaniesWorked  

---

## **Dependencies**

This project uses the following libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **scikit-learn**: For preprocessing (StandardScaler and OneHotEncoder) and train-test split.
- **TensorFlow/Keras**: For building, training, and evaluating the neural network.

### Install Dependencies
To install the required libraries, run:
```bash
pip install pandas numpy scikit-learn tensorflow
```

---

## **Code Walkthrough**
Part 1: Preprocessing

	1.	Data Import:
	•	Load the dataset into a Pandas DataFrame.
	•	Inspect the first five rows and unique values for each column.
	2.	Target and Feature Selection:
	•	Attrition and Department as targets.
	•	Selected 10 features for the model.
	3.	Data Encoding:
	•	OneHotEncoding for categorical features (OverTime, Attrition, Department).
	4.	Feature Scaling:
	•	StandardScaler to normalize numeric features.
	5.	Train-Test Split:
	•	Split the dataset into 75% training and 25% testing data.

Part 2: Neural Network Model

	1.	Model Design:
	•	Input Layer: Takes 11 input features.
	•	Shared Layers: Two dense layers with 64 and 128 units, ReLU activation.
	•	Output Branches:
	•	Department Branch: Dense layer with 32 units, output layer with softmax activation (3 classes).
	•	Attrition Branch: Dense layer with 32 units, output layer with sigmoid activation (binary classification).
	2.	Model Compilation:
	•	Optimizer: Adam.
	•	Loss Functions: categorical_crossentropy (Department), binary_crossentropy (Attrition).
	•	Metrics: Accuracy for both branches.
	3.	Training:
	•	Trained with 100 epochs, batch size of 32, and validation data.
	4.	Evaluation:
	•	Evaluated accuracy and loss for both output branches.

---

## **Summary**

### Model Summary
	•	Two output branches for predicting Attrition and Department.
	•	Total Parameters: 17,445.

### Evaluation Metrics
	•	Department Accuracy: 52.7%
	•	Attrition Accuracy: 82.6%

---

## **Usage**
	1.	Clone the repository
    2.	Install dependencies.
	3.	Open attrition.ipynb in Jupyter Notebook or Google Colab.
	4.	Run the notebook cells step-by-step to reproduce the results.

## Author's Environment Details

### Environment Details
- **Python Implementation**: CPython
- **Python Version**: 3.10.14
- **IPython Version**: 8.25.0
- **Compiler**: Clang 14.0.6
- **Operating System**: Darwin
- **Release**: 23.4.0
- **Machine**: arm64
- **Processor**: arm
- **CPU Cores**: 8
- **Architecture**: 64-bit

### Installed Packages
- **tensorflow**: 2.18.0
- **keras**: 3.7.0
- **sklearn**: 1.4.2
- **requests**: 2.32.2
- **watermark**: 2.5.0
- **IPython**: 8.25.0
- **ipywidgets**: 8.1.5
- **numpy**: 1.26.4
- **json**: 2.0.9
- **xarray**: 2023.6.0
- **pandas**: 2.2.2

### System Information
- **sys**: 3.10.14 (main, May 6 2024, 14:42:37) [Clang 14.0.6]

---
## **Challenge Instructions**

Background
You are tasked with creating a neural network that HR can use to predict whether employees are likely to leave the company. Additionally, HR believes that some employees may be better suited to other departments, so you are also asked to predict the department that best fits each employee. These two columns should be predicted using a branched neural network.
Files
Download the following files to help you get started:
Module 19 Challenge filesLinks to an external site.
Before You Begin
Before starting the Challenge, be sure to complete the following steps:
- Create a new repository for this project called neural-network-challenge-2. Do not add this Challenge assignment to an existing repository.
- Clone the new repository to your computer.
- Inside your local Git repository, add the starter file attrition.ipynb from your file downloads.
- Push these changes to GitHub.
- Upload attrition.ipynb to Google Colab and work on your solution there.
- Make sure to periodically download your file and push the changes to your repository.

Instructions
Open the starter file in Google Colab and complete the following steps, which are divided into three parts:
Part 1: Preprocessing
1. Import the data and view the first five rows.
2. Determine the number of unique values in each column.
3. Create y_df with the attrition and department columns.
4. Create a list of at least 10 column names to use as X data. You can choose any 10 columns you’d like EXCEPT the attrition and department columns.
5. Create X_df using your selected columns.
6. Show the data types for X_df.
7. Split the data into training and testing sets.
8. Convert your X data to numeric data types however you see fit. Add new code cells as necessary. Make sure to fit any encoders to the training data, and then transform both the training and testing data.
9. Create a StandardScaler, fit the scaler to the training data, and then transform both the training and testing data.
10. Create a OneHotEncoder for the department column, then fit the encoder to the training data and use it to transform both the training and testing data.
11. Create a OneHotEncoder for the attrition column, then fit the encoder to the training data and use it to transform both the training and testing data.
Part 2: Create, Compile, and Train the Model
1. Find the number of columns in the X training data.
2. Create the input layer. Do NOT use a sequential model, as there will be two branched output layers.
3. Create at least two shared layers.
4. Create a branch to predict the department target column. Use one hidden layer and one output layer.
5. Create a branch to predict the attrition target column. Use one hidden layer and one output layer.
6. Create the model.
7. Compile the model.
8. Summarize the model.
9. Train the model using the preprocessed data.
10. Evaluate the model with the testing data.
11. Print the accuracy for both department and attrition.
Part 3: Summary
Briefly answer the following questions in the space provided:
1. Is accuracy the best metric to use on this data? Why or why not?
2. What activation functions did you choose for your output layers, and why?
3. Can you name a few ways that this model could be improved?

Hints and Considerations
- Review previous modules if you need help with data preprocessing.
- Make certain that your training and testing data are preprocessed in the same ways.
- Review Day 3 of this module for information on branching neural networks.

Requirements
Preprocessing (40 points)
- Import the data. (5 points)
- Create y_df with the attrition and department columns. (5 points)
- Choose 10 columns for X. (5 points)
- Show the data types of the X columns. (5 points)
- Split the data into training and testing sets. (5 points)
- Encode all X data to numeric types. (5 points)
- Scale the X data. (5 points)
- Encode all y data to numeric types. (5 points)
Model (40 points)
- Find the number of columns in the X training data. (5 points)
- Create an input layer. (5 points)
- Create at least two shared hidden layers. (10 points)
- Create an output branch for the department column. (10 points)
- Create an output branch for the attrition column. (10 points)
Summary (20 points)
- Answer the questions briefly. (10 points)
- Show understanding of the concepts in your answers. (10 points)

