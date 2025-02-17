
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#%%
def train_model ( X_train , y_train , classifier_type ):
    if classifier_type == 'logistic':
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif classifier_type == 'svm':
        model = SVC(kernel='linear', class_weight='balanced')
    elif classifier_type == 'naive_bayes':
        model = MultinomialNB()
    model.fit(X_train, y_train)
        
#%%

