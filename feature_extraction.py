from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#%% extract features in dataset
def extract_features(data):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True)
    X = vectorizer.fit_transform(data['text'])
    y = data['rating']
    return X, y, vectorizer

#%% split dataset into training, validation and testing sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

#%%