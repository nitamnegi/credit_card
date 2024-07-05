# Credit Card Fraud Detection

This project demonstrates the detection of credit card fraud using machine learning techniques. The dataset used is the publicly available credit card transaction dataset which contains transactions made by credit cards in September 2013 by European cardholders. 

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud detection is critical for preventing financial losses and protecting cardholders. This project uses machine learning models to identify fraudulent transactions from a dataset of credit card transactions.

## Dataset

The dataset used in this project is `creditcard.csv`, which contains the following features:
- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: The principal components obtained with PCA.
- `Amount`: The transaction amount.
- `Class`: The label where 1 indicates fraud and 0 indicates normal transactions.

- Link to dataset: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Installation

To run this project, you will need Python and the following libraries:

- pandas
- scikit-learn
- numpy
- seaborn
- matplotlib

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn numpy seaborn matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/chyash1110/credit_card_fraud_detection.git
```

2. Navigate to the project directory:

```bash
cd credit_card_fraud_detection
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook
```

4. Open `Credit Card Fraud Detection.ipynb` and execute the cells to preprocess the data, train the model, and evaluate the results.

## Modeling

The project uses a Support Vector Machine (SVM) for modeling. The steps include:

1. Data preprocessing: Standardizing the features and splitting the dataset into training and testing sets.
2. Model training: Training an SVM model on the training data.
3. Model evaluation: Evaluating the model on the test data.

## Evaluation

The model is evaluated using the following metrics:

- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Results

The results of the model, including the confusion matrix and classification report, are visualized using Seaborn and printed out in the notebook.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
