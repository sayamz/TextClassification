# data_loader.py - Handles loading the dataset

import pandas as pd

#%%
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['reviews.text', 'reviews.rating']].dropna()
    data.columns = ['text', 'rating']
    return data

#%%