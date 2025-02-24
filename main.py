from data_loader import load_data
from preprocessing import preprocess_data #, preprocess_data_new,preprocess_data_1
from feature_extraction import extract_features, split_data
from qualitative_analysis import qualitative_analysis
from model_training import train_model
from evaluation import evaluate_model

#%% Entry point of the classifier
def main():
    dataset_path = 'Dataset/amazon_reviews.csv'
    data = load_data(dataset_path)
    data = preprocess_data(data, method='stemming')
    X, y = extract_features(data, use_tfidf=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    models = {
        'logistic': train_model(X_train, y_train, 'logistic'),
        'svm': train_model(X_train, y_train, 'svm'),
        'naive_bayes': train_model(X_train, y_train, 'naive_bayes')
    }
    
    for name, model in models.items():
        print(f"\nEvaluating {name} model:")
        evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Perform qualitative analysis on best-performing model
    print("\n### Qualitative Analysis ###\n")
    
    # Ensure we pass only the reviews corresponding to the test set
    test_reviews = data.iloc[y_test.index]["text"].tolist()  # Select reviews for test set only

    qualitative_analysis(test_reviews, y_test.tolist(), models["naive_bayes"].predict(X_test).tolist()) #  SVM is my best model    

    print("\nExperiment Complete!")

if __name__ == "__main__":
    main()

#%%