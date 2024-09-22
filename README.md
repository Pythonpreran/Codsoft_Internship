# Machine Learning Internship Projects

This repository contains the projects completed during my Machine Learning internship. Each task focuses on a different aspect of machine learning and data analysis.

## **Task 1: Movie Genre Detector Using Plot**

In this project, I developed a machine learning model to classify movie genres based on their plot descriptions. The model utilizes TF-IDF vectorization and a Logistic Regression algorithm to predict the genre of a movie from its plot.

### **Output**
- **Train Data Sample**: Displays a sample of the training data used for the model.
- **Predicted Genre**: Given a plot input, the model predicts the genre.

### **Example**
- **Input**: A man returns home, only to find that his house is haunted and evil forces try to kill him.
- **Predicted Genre**: Horror

## **Task 2: Subscription Churn Model**

This task involved creating a predictive model to identify customers who are likely to cancel their subscriptions. The project included:

- Loading and preprocessing the dataset.
- Encoding categorical variables and splitting the data into training and testing sets.
- Training multiple models, including Logistic Regression, Random Forest, and Gradient Boosting.
- Evaluating the models based on confusion matrix and classification report.

### **Output**
- Performance metrics for each model, highlighting accuracy and other relevant statistics.

## **Task 3: SPAM Message Detector Using Support Vector Machines**

In this project, I implemented a spam detection system using Support Vector Machines (SVM). The model was trained on a dataset of messages labeled as spam or legitimate.

### **Process**
- The dataset was preprocessed to focus on relevant features.
- Text data was vectorized using TF-IDF.
- The SVM model was trained and evaluated on its ability to classify messages accurately.

### **Output**
- SVM model accuracy and classification report.
- Functionality to classify user-input messages as SPAM or LEGITIMATE.

## **Installation**

To run the code in this repository, please ensure you have the following Python libraries installed:

```bash
pip install Flask pandas statsmodels scikit-learn xgboost matplotlib
