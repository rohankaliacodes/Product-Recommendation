# Importing necessary libraries
import pandas as pd  # For data manipulation and analysis
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer  # For converting text data into a matrix of token counts
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors

# This function determines whether we're running as an executable (frozen) or as a script
def get_base_dir():
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app 
        # path into variable _MEIPASS'.
        return sys._MEIPASS
    else:
        return os.path.dirname(__file__)

# Function to load data from a CSV file
def load_data(file_name):
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, file_name)
    try:
        # Attempt to read the CSV file into a DataFrame
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        # If the file is not found, return an error message
        return f"File not found at {file_path}. Please check the file path."

# Function to validate and preprocess the input data
def preprocess_data(grocery_data):
    # Define the required columns for the recommendation system
    required_columns = ['Order ID', 'Customer Name', 'Category', 'Sub Category']
    
    # Check if all required columns are present in the data
    if not all(col in grocery_data.columns for col in required_columns):
        # If not, return an error message
        return "Data missing required columns."

    # Remove rows with missing values in the required columns
    grocery_data.dropna(subset=required_columns, inplace=True)

    # Identify any columns in the data that are not required
    unnecessary_columns = set(grocery_data.columns) - set(required_columns)
    # Remove these unnecessary columns from the DataFrame
    grocery_data.drop(columns=unnecessary_columns, inplace=True, errors='ignore')

    # Group the data by Customer Name, Category, and Sub Category, and count the occurrences to create customer profiles
    customer_profiles = grocery_data.groupby(['Customer Name', 'Category', 'Sub Category']).size().reset_index(name='Purchase Count')
    # Create a feature representation of each customer's profile by concatenating the category and subcategory, weighted by purchase count
    customer_profiles['Features'] = customer_profiles.apply(lambda x: (x['Category'] + ' ' + x['Sub Category'] + ' ') * x['Purchase Count'], axis=1)
    # Aggregate these features for each customer into a single string
    aggregated_profiles = customer_profiles.groupby('Customer Name')['Features'].sum().reset_index()
    # Return both the detailed customer profiles and the aggregated profile information
    return aggregated_profiles, customer_profiles

# Function to generate product recommendations for a given customer
def recommend_products(aggregated_profiles, customer_profiles, customer_name, top_n=5):
    # Initialize the CountVectorizer to convert the aggregated profile strings into a matrix of token counts
    vectorizer = CountVectorizer()
    # Fit the vectorizer to the aggregated profiles and transform the data
    count_matrix = vectorizer.fit_transform(aggregated_profiles['Features'])
    
    # Check if the given customer name exists in the aggregated profiles
    if customer_name not in aggregated_profiles['Customer Name'].values:
        # If not, return an error message
        return "Customer not found."

    # Find the index of the customer in the DataFrame
    idx = aggregated_profiles.index[aggregated_profiles['Customer Name'] == customer_name].tolist()[0]
    # Calculate the cosine similarity between the customer's profile and all other profiles
    cosine_sim = cosine_similarity(count_matrix)
    # Create a list of tuples pairing customer indices with their similarity scores
    similarity_scores = list(enumerate(cosine_sim[idx]))
    # Sort the list of tuples by similarity score in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    # Identify the indices of the top N most similar customers
    top_customers_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    
    # Find the categories and sub-categories purchased by these top similar customers, avoiding duplicates
    recommended_categories = customer_profiles[customer_profiles['Customer Name'].isin(aggregated_profiles.iloc[top_customers_indices]['Customer Name'])].drop_duplicates()
    # Count how many times each category and sub-category appears and sort by this count
    recommendations = recommended_categories.groupby(['Category', 'Sub Category']).size().reset_index(name='Counts')
    # Sort the recommendations based on the count, in descending order, and take the top N results
    recommendations = recommendations.sort_values(by='Counts', ascending=False).head(top_n)
    
    # Format the recommendations into a string for readability, including the category, sub-category, and count
    formatted_recommendations = ""
    for index, row in recommendations.iterrows():
        formatted_recommendations += f"- {row['Category']} > {row['Sub Category']} (recommended based on {row['Counts']} similar purchases)\n"
    
    # Return a formatted string with a message and the list of recommendations
    return f"Based on your purchase history and similar customers' preferences, we recommend the following products:\n{formatted_recommendations}"

# Main execution example
# Set the file name for the CSV data
file_name = 'grocery_sells.csv'
# Load the data using the load_data function
grocery_data = load_data(file_name)
# Check if the data was loaded successfully as a DataFrame
if isinstance(grocery_data, pd.DataFrame):
    # Preprocess the data to create customer profiles
    aggregated_profiles, customer_profiles = preprocess_data(grocery_data)
    # Check if preprocessing was successful
    if isinstance(aggregated_profiles, pd.DataFrame):
        # Select a test customer from the dataset
        test_customer = aggregated_profiles['Customer Name'][0]
        # Print the product recommendations for the test customer
        print(recommend_products(aggregated_profiles, customer_profiles, test_customer))
    else:
        # If there was an error during preprocessing, print the error message
        print(aggregated_profiles)
else:
    # If there was an error loading the data, print the error message
    print(grocery_data)
