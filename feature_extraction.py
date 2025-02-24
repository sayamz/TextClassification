from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

#%% extract features in dataset
def extract_features(data, use_tfidf=False):
    if use_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True)
    else:
        vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True)
        
    X = vectorizer.fit_transform(data['text'])
    y = data['rating']
    return X, y # vectorizer

#%% split dataset into training, validation and testing sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42) # split dataset in the ratio 60%/20%/20% respectively for train/validation/test. Train here is taking 60%
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) # now split the remaining 40% equally between validation and test
    return X_train, X_val, X_test, y_train, y_val, y_test

#%%