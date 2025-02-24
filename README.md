This project files is structured as follows:

textclassification/
│── main.py               # Entry point to run the full pipeline
│── data_loader.py        # Loads dataset from CSV
│── preprocessing.py      # Cleans text (lowercasing, stopwords, stemming/lemmatization)
│── feature_extraction.py # Converts text to numerical features (TF-IDF/CountVectorizer)
│── model_training.py     # Trains models (Logistic Regression, SVM, Naïve Bayes)
│── evaluation.py         # Evaluates models using accuracy, confusion matrix
│── qualitative_analysis.py # Analyzes correct/misclassified reviews
│── amazon_reviews.csv    # Dataset (ensure this is present in the directory)
│── README.txt            # Instructions
│── requirements.txt      # List of required libraries
|── Dataset/amazon_reviews.csv #contains the dataset used for the experiments
|── Results              # This directory contains the results of the experiments performed

How to Run the Project (on bash)
=======================
python main.py


Customization
=============
To Change Preprocessing Method (Stemming vs. Lemmatization): 
Edit main.py and on line 12, change the parameter value for method as shown below: 
data = preprocess_data(data, method="stemming")  # Change to "lemmatization" if needed

Switch Feature Extraction Method (TF-IDF vs. CountVectorizer)
=============================================================
Edit main.py and on line 13, change the parameter value for use_tfidf as shown below: 
X, y, vectorizer = extract_features(data, use_tfidf=True)  # Set False for CountVectorizer


