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
from nltk.corpus import wordnet

#%%
# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
dataset_path = 'Dataset/amazon_reviews.csv'
data = pd.read_csv(dataset_path)
data = data[['reviews.text', 'reviews.rating']].dropna()
data.columns = ['text', 'rating']
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
        words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]
    
    return ' '.join(words)

#%% Apply preprocessing
data['text'] = data['text'].apply(lambda x: preprocess_text(x, method='lemmatization'))

#%% Feature Extraction
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True)
X = vectorizer.fit_transform(data['text'])
y = data['rating']

#%% Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Train models
def train_model(X_train, y_train, classifier_type):
    if classifier_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif classifier_type == 'svm':
        model = SVC(kernel='linear', class_weight='balanced')
    elif classifier_type == 'naive_bayes':
        model = MultinomialNB()
    
    model.fit(X_train, y_train)
    return model

#%% Train and evaluate models
logistic_model = train_model(X_train, y_train, 'logistic')
svm_model = train_model(X_train, y_train, 'svm')
naive_bayes_model = train_model(X_train, y_train, 'naive_bayes')

#%% Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

#print("\nEvaluating Logistic Regression model:")
#evaluate_model(logistic_model, X_test, y_test)

print("\nEvaluating SVM model:")
evaluate_model(svm_model, X_test, y_test)

#print("\nEvaluating Naive Bayes model:")
#evaluate_model(naive_bayes_model, X_test, y_test)

print("\nExperiment Complete!")
#%%