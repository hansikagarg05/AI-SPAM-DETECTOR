# AI-Based Spam Message Detection System

## Overview

The AI-Based Spam Message Detection System is a machine learning project
that classifies SMS messages as either SPAM or HAM (legitimate).

The system uses text preprocessing, TF-IDF feature extraction, and a
Multinomial Naive Bayes classifier to analyze messages and predict their
category.

## Features

- SMS text preprocessing
- TF-IDF feature extraction
- Multinomial Naive Bayes classification
- SPAM/HAM prediction
- Prediction confidence
- Model accuracy and classification report
- Automated testing using pytest

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Pytest

## Project Workflow

Input SMS
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Naive Bayes Classifier
↓
SPAM / HAM Prediction
↓
Confidence Score

## Dataset

The project uses the SMS Spam Collection dataset.

The dataset contains labelled SMS messages classified as either
`spam` or `ham`.

## Project Structure

AI-SPAM-DETECTOR/
│
├── data/
├── src/
├── tests/
├── main.py
├── requirements.txt
├── statement.md
├── pytest.ini
├── .gitignore
└── README.md

## Installation

Create and activate a Python virtual environment and install the required
libraries using:

pip install -r requirements.txt

## How to Run

Run the main program using:

python main.py

Enter an SMS message when prompted to get the prediction.

## Testing

Run the automated tests using:

pytest

## Model Result

The current implementation achieved approximately 95.78% accuracy on the
test set.

## Future Enhancements

- Improve spam recall
- Add a graphical user interface
- Add more datasets for training
- Compare multiple machine learning algorithms
- Deploy the system as a web application
## Screenshots

### HAM Message Detection

![HAM Detection](screenshots/ham_result.png)

### SPAM Message Detection

![SPAM Detection](screenshots/spam_result.png)-