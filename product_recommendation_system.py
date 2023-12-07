import pandas as pd  # For data manipulation and analysis
from sklearn.feature_extraction.text import CountVectorizer  # For converting text data into a matrix of token counts
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        return "File not found. Please check the file path."
