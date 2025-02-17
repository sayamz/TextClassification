from data_loader import load_data
from preprocessing import preprocess_data
from feature_extraction import extract_features, split_data
from model_training import train_model
from evaluation import evaluate_model

#%% Entry point of the classifier
def main():
    dataset_path = 'Dataset/amazon_reviews.csv'
    data = load_data(dataset_path)
    data = preprocess_data(data, method='stemming')
    X, y, vectorizer = extract_features(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    models = {
        #'logistic': train_model(X_train, y_train, 'logistic'),
        'svm': train_model(X_train, y_train, 'svm'),
        #'naive_bayes': train_model(X_train, y_train, 'naive_bayes')
    }
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model:")
        evaluate_model(model, X_val, y_val, X_test, y_test)
    
    print("\nExperiment Complete!")

if __name__ == "__main__":
    main()

#%%