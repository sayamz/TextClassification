import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
#%%

# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

#%%
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['reviews.text', 'reviews.rating']].dropna()
    data.columns = ['text', 'rating']
    return data

#%%
def preprocess_text(text, method='stemming'):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    if method == 'stemming':
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    elif method == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)
#%%

def preprocess_data(data, method='stemming'):
    data['text'] = data['text'].apply(lambda x: preprocess_text(x, method))
    return data

#%%
def extract_features(data):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True)
    X = vectorizer.fit_transform(data['text'])
    y = data['rating']
    return X, y, vectorizer

#%%
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

#%%
def train_model(X_train, y_train, classifier_type):
    if classifier_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif classifier_type == 'svm':
        model = SVC(kernel='linear', class_weight='balanced')
    elif classifier_type == 'naive_bayes':
        model = MultinomialNB()
    
    model.fit(X_train, y_train)
    return model
#%%

def evaluate_model(model, X_val, y_val, X_test, y_test):
    print("Validation Set Performance:")
    val_predictions = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_val, val_predictions))
    print("Classification Report:\n", classification_report(y_val, val_predictions))
    
    print("\nTest Set Performance:")
    test_predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, test_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_predictions))
    print("Classification Report:\n", classification_report(y_test, test_predictions))

#%%

def main():
    dataset_path = 'Dataset/amazon_reviews.csv'
    data = load_data(dataset_path)
    data = preprocess_data(data, method='lemmatization')
    X, y, vectorizer = extract_features(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    models = {
        'logistic': train_model(X_train, y_train, 'logistic'),
        'svm': train_model(X_train, y_train, 'svm'),
        'naive_bayes': train_model(X_train, y_train, 'naive_bayes')
    }
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model:")
        evaluate_model(model, X_val, y_val, X_test, y_test)
    
    print("\nExperiment Complete!")

#%%

if __name__ == "__main__":
    main()

#%%