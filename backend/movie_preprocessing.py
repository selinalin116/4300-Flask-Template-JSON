import os
import re
import string
from collections import Counter
import helperfunctions

def load_food_database(file_path):
    """
    Load food names from a database file.
    Each line in the file should contain one food name.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            food_names = [line.strip() for line in file.readlines()]
        return food_names
    except Exception as e:
        print(f"Error loading food database: {e}")
        return []

# def find_script_file(movie_title, script_folder):
#     """
#     Find the script file for a given movie title.
#     Handles various formats including spaces vs. hyphens.
#     """
#     movie_title_lower = movie_title.lower()
    
#     # Create variations of the movie title for matching
#     variations = [
#         movie_title_lower,                      # Original (e.g., "star wars")
#         movie_title_lower.replace(' ', '-'),    # Spaces to hyphens (e.g., "star-wars")
#         movie_title_lower.replace(' ', '_'),    # Spaces to underscores (e.g., "star_wars")
#         movie_title_lower.replace('-', ' '),    # Hyphens to spaces
#         movie_title_lower.replace('_', ' '),    # Underscores to spaces
#         ''.join(movie_title_lower.split())      # No spaces (e.g., "starwars")
#     ]
    
#     # Try exact match first
#     for filename in os.listdir(script_folder):
#         if filename.endswith('.txt'):
#             filename_without_ext = os.path.splitext(filename)[0].lower()
            
#             # Check if any variation matches the filename
#             for variation in variations:
#                 if filename_without_ext == variation:
#                     return os.path.join(script_folder, filename)
    
#     # Try partial match if exact match fails
#     for filename in os.listdir(script_folder):
#         if filename.endswith('.txt'):
#             filename_without_ext = os.path.splitext(filename)[0].lower()
            
#             # Check if any variation is contained in the filename
#             for variation in variations:
#                 if variation in filename_without_ext:
#                     return os.path.join(script_folder, filename)
    
#     # If no match is found
#     return None

def process_script_for_food(script_text, food_names):
    """
    Process a script's text content and find food mentions.
    """
    try:
        # Preprocess the script
        script_text = script_text.lower()
        script_text = re.sub(f'[{string.punctuation}]', ' ', script_text)
        script_text = re.sub(r'\s+', ' ', script_text).strip()
        
        # Find food mentions
        food_mentions = Counter()
        for food in food_names:
            food_lower = food.lower()
            pattern = r'\b' + re.escape(food_lower) + r'\b'
            matches = re.findall(pattern, script_text)
            if matches:
                food_mentions[food] = len(matches)
        
        return food_mentions, script_text
    
    except Exception as e:
        print(f"Error processing script: {e}")
        return Counter(), ""

def get_movie_foods(movie_title, script_folder, food_database_path):
    """
    Main function to find food mentions in a movie script.
    Returns a dictionary with just the movie title and foods.
    """
    # Load food database
    food_names = load_food_database(food_database_path)
    if not food_names:
        return {"error": "Food database could not be loaded"}
    
    # Find the script file
    script_path = helperfunctions.get_movie_script(movie_title, script_folder)
    if not script_path:
        return {
            "movie_title": movie_title,
            "error": f"No script found for movie '{movie_title}'."
        }
    
    # Get the actual movie title from the filename
    movie_name = os.path.splitext(os.path.basename(script_path))[0]
    
    # Process the script for food mentions
    food_mentions, preprocessed_text = process_script_for_food(script_path, food_names)
    
    if not food_mentions:
        return {
            "movie_title": movie_name,
            "foods": []
        }
    
    # Return movie title and foods with counts
    return {
        "movie_title": movie_name,
        "foods": [{
            "name": food,
            "count": count
        } for food, count in food_mentions.most_common()]
    }

# import os
# import re
# import string
# import numpy as np
# import pandas as pd
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.sparse.linalg import svds
# from sklearn.preprocessing import normalize

# vectorizer = None
# td_matrix = None
# docs_compressed = None
# docs_compressed_normed = None
# words_compressed = None
# words_compressed_normed = None
# index_to_title = {}
# word_to_index = {}
# index_to_word = {}
# food_names = []

# def load_food_database(food_database_path):
#     """Load food names from the CSV database file with a single column"""
#     try:
#         # Read the CSV file containing food names
#         food_df = pd.read_csv(food_database_path, header=None)
        
#         # Since there's only one column with name, use the first column
#         foods = food_df[0].astype(str).str.lower().str.strip().tolist()
        
#         # Remove duplicates and empty strings
#         foods = list(set([name for name in foods if name and name != 'nan']))
        
#         return foods
#     except Exception as e:
#         print(f"Error loading food database: {e}")
#         return []
    
