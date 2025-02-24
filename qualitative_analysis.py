import pandas as pd

#%% this method analyses correct & incorrect predictions
def qualitative_analysis(reviews, actual_ratings, predicted_ratings, num_samples=5):
    
    # Create a DataFrame with actual vs predicted ratings
    df = pd.DataFrame({'Review': reviews, 'Actual': actual_ratings, 'Predicted': predicted_ratings})

    # Separate correctly and incorrectly classified reviews
    correct_preds = df[df["Actual"] == df["Predicted"]]
    incorrect_preds = df[df["Actual"] != df["Predicted"]]

    # Sample a few correctly and incorrectly classified reviews
    correct_samples = correct_preds.sample(min(num_samples, len(correct_preds)), random_state=42)
    incorrect_samples = incorrect_preds.sample(min(num_samples, len(incorrect_preds)), random_state=42)

    # Display results
    print("\n### Correctly Classified Examples ###")
    print(correct_samples.to_string(index=False))

    print("\n### Incorrectly Classified Examples ###")
    print(incorrect_samples.to_string(index=False))
