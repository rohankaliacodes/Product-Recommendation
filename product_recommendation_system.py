import pandas as pd  # For data manipulation and analysis
from sklearn.feature_extraction.text import CountVectorizer  # For converting text data into a matrix of token counts
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        return "File not found. Please check the file path."
    
def preprocess_data(grocery_data):
    required_columns = ['Order ID', 'Customer Name', 'Category', 'Sub Category']
    if not all(col in grocery_data.columns for col in required_columns):
        return "Data missing required columns."
    grocery_data.dropna(subset=required_columns, inplace=True)
    unnecessary_columns = set(grocery_data.columns) - set(required_columns)
    grocery_data.drop(columns=unnecessary_columns, inplace=True, errors='ignore')
    customer_profiles = grocery_data.groupby(['Customer Name', 'Category', 'Sub Category']).size().reset_index(name='Purchase Count')
    customer_profiles['Features'] = customer_profiles.apply(lambda x: (x['Category'] + ' ' + x['Sub Category'] + ' ') * x['Purchase Count'], axis=1)
    aggregated_profiles = customer_profiles.groupby('Customer Name')['Features'].sum().reset_index()
    return aggregated_profiles, customer_profiles
