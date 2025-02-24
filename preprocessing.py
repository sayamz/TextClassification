import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

#%% Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

#%%
def preprocess_text(text, method=None):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize words
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Apply stemming or lemmatization if specified
    if method == 'stemming':
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    elif method == 'lemmatization':
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

    # Return processed text as a single string
    return ' '.join(words)

def preprocess_data(data, method=None):
    data['text'] = data['text'].apply(lambda x: preprocess_text(x, method))
    return data

#%%