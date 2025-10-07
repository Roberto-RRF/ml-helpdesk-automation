# ml-helpdesk-automation

A comparative study of machine learning models for IT support ticket classification.

## Overview

This project compares five different machine learning approaches to automatically classify IT support tickets as either Hardware or Non-Hardware related:

- **Logistic Regression** - 87.7% accuracy
- **Support Vector Machine (SVM)** - 87.0% accuracy  
- **Random Forest** - 85.4% accuracy
- **LSTM Neural Network** - 81.1% accuracy
- **CNN Neural Network** - 81.1% accuracy

## Key Finding

Traditional machine learning models (Logistic Regression, SVM) outperformed deep learning approaches by 6-7% on a dataset of 5,000 IT tickets. This suggests that for moderate-sized datasets and binary classification tasks, simpler models are more effective and computationally efficient.

## Dataset

- **Source**: IT Service Ticket Dataset
- **Size**: 5,000 tickets (balanced from original 47,837)
- **Classes**: Hardware (2,500) vs Non-Hardware (2,500)

## Methodology

1. **Text Preprocessing**: Lowercasing, contraction expansion, tokenization, stop word removal, lemmatization
2. **Feature Extraction**: TF-IDF (traditional ML) and Word2Vec (deep learning)
3. **Model Training**: 80/20 train-test split with stratification
4. **Evaluation**: Accuracy, Precision, Recall, F1-Score

## Requirements
```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib seaborn
