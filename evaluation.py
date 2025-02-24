from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nTraining Set Performance:")
    train_predictions = model.predict(X_train)
    print("Accuracy:", accuracy_score(y_train, train_predictions))
    
    print("\nValidation Set Performance:")
    val_predictions = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, val_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_val, val_predictions))
    print("Classification Report:\n", classification_report(y_val, val_predictions, zero_division=1))

    print("\nTest Set Performance:")
    test_predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, test_predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_predictions))
    print("Classification Report:\n", classification_report(y_test, test_predictions, zero_division=1))