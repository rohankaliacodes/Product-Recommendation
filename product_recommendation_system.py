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

def recommend_products(aggregated_profiles, customer_profiles, customer_name, top_n=5):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(aggregated_profiles['Features'])
    if customer_name not in aggregated_profiles['Customer Name'].values:
        return "Customer not found."
    idx = aggregated_profiles.index[aggregated_profiles['Customer Name'] == customer_name].tolist()[0]
    cosine_sim = cosine_similarity(count_matrix)
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_customers_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    recommended_categories = customer_profiles[customer_profiles['Customer Name'].isin(aggregated_profiles.iloc[top_customers_indices]['Customer Name'])].drop_duplicates()
    recommendations = recommended_categories.groupby(['Category', 'Sub Category']).size().reset_index(name='Counts')
    recommendations = recommendations.sort_values(by='Counts', ascending=False).head(top_n)
    formatted_recommendations = ""
    for index, row in recommendations.iterrows():
        formatted_recommendations += f"- {row['Category']} > {row['Sub Category']} (recommended based on {row['Counts']} similar purchases)\n"
    return f"Based on your purchase history and similar customers' preferences, we recommend the following products:\n{formatted_recommendations}"
