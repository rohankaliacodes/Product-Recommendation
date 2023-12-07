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
    required_columns = ['Customer Name', 'Category', 'Sub Category']
    
    if not all(col in grocery_data.columns for col in required_columns):
        return "Data missing required columns."

    grocery_data.dropna(subset=required_columns, inplace=True)

    # Replace spaces with underscores in categories and subcategories
    grocery_data['Category'] = grocery_data['Category'].str.replace(' ', '_')
    grocery_data['Sub Category'] = grocery_data['Sub Category'].str.replace(' ', '_')

    # Group data and count occurrences to create customer profiles
    customer_profiles = grocery_data.groupby(['Customer Name', 'Category', 'Sub Category']).size().reset_index(name='Purchase Count')
    customer_profiles['Features'] = customer_profiles.apply(lambda x: (x['Category'] + '_' + x['Sub Category'] + ' ') * x['Purchase Count'], axis=1)
    
    # Aggregate features for each customer into a single string
    aggregated_profiles = customer_profiles.groupby('Customer Name')['Features'].sum().reset_index()

    return aggregated_profiles



# Function to generate product recommendations for a given customer

def recommend_products(aggregated_profiles, customer_name, top_n=5):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(aggregated_profiles['Features'])
    
    if customer_name not in aggregated_profiles['Customer Name'].values:
        return "Customer not found."

    idx = aggregated_profiles.index[aggregated_profiles['Customer Name'] == customer_name].tolist()[0]
    cosine_sim = cosine_similarity(count_matrix)

    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Use actual similarity scores in the recommendation process
    recommendations = {}
    for i, score in similarity_scores[1:top_n+1]:
        similar_customer_features = aggregated_profiles.iloc[i]['Features'].split()
        for feature in similar_customer_features:
            if feature in recommendations:
                recommendations[feature] += score
            else:
                recommendations[feature] = score

    # Sort recommendations by their scores
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Format the recommendations for readability
    formatted_recommendations = "\n".join([f"- {prod.replace('_', ' ')} (score: {round(score, 2)})" for prod, score in sorted_recommendations])

    return f"Based on your purchase history and similar customers' preferences, we recommend the following products:\n{formatted_recommendations}"

# Main execution example
# Main execution example
if __name__ == '__main__':
    # Set the file name for the CSV data
    file_name = 'grocery_sells.csv'
    # Load the data using the load_data function
    grocery_data = load_data(file_name)
    
    # Check if the data was loaded successfully as a DataFrame
    if isinstance(grocery_data, pd.DataFrame):
        # Preprocess the data to create aggregated customer profiles
        aggregated_profiles = preprocess_data(grocery_data)
        
        # Check if preprocessing was successful
        if isinstance(aggregated_profiles, pd.DataFrame):
            # Prompt for the customer's name to provide personalized recommendations
            customer_name_input = input("Please enter your name for personalized product recommendations: ").strip().capitalize()
            # Similarly, convert customer names in the dataset to lower case during preprocessing

            # Print the product recommendations for the input customer name
            print(recommend_products(aggregated_profiles, customer_name_input))
            input("Click enter to end")
        else:
            # If there was an error during preprocessing, print the error message
            print(aggregated_profiles)
    else:
        # If there was an error loading the data, print the error message
        print(grocery_data)
