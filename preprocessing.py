import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

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