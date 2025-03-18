import os
import re
import string
from collections import Counter

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

def find_script_file(movie_title, script_folder):
    """
    Find the script file for a given movie title.
    Handles various formats including spaces vs. hyphens.
    """
    movie_title_lower = movie_title.lower()
    
    # Create variations of the movie title for matching
    variations = [
        movie_title_lower,                      # Original (e.g., "star wars")
        movie_title_lower.replace(' ', '-'),    # Spaces to hyphens (e.g., "star-wars")
        movie_title_lower.replace(' ', '_'),    # Spaces to underscores (e.g., "star_wars")
        movie_title_lower.replace('-', ' '),    # Hyphens to spaces
        movie_title_lower.replace('_', ' '),    # Underscores to spaces
        ''.join(movie_title_lower.split())      # No spaces (e.g., "starwars")
    ]
    
    # Try exact match first
    for filename in os.listdir(script_folder):
        if filename.endswith('.txt'):
            filename_without_ext = os.path.splitext(filename)[0].lower()
            
            # Check if any variation matches the filename
            for variation in variations:
                if filename_without_ext == variation:
                    return os.path.join(script_folder, filename)
    
    # Try partial match if exact match fails
    for filename in os.listdir(script_folder):
        if filename.endswith('.txt'):
            filename_without_ext = os.path.splitext(filename)[0].lower()
            
            # Check if any variation is contained in the filename
            for variation in variations:
                if variation in filename_without_ext:
                    return os.path.join(script_folder, filename)
    
    # If no match is found
    return None

def process_script_for_food(script_path, food_names):
    """
    Process a single script file and find food mentions from the food database.
    """
    try:
        # Read the script
        with open(script_path, 'r', encoding='utf-8') as file:
            script_text = file.read()
        
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

# def extract_food_contexts(script_text, food_names, food_mentions, context_words=5):
#     """
#     Extract context for food mentions in the script.
#     """

#     # Dictionary to store food names (key) and its surrounding context (values)
#     food_contexts = {}
    
#     try:
#         # Split split into words
#         words = script_text.split()
        
#         # Find contexts for each food
#         for food in food_names:
#             if food not in food_mentions or food_mentions[food] == 0:
#                 continue
                
#             food_lower = food.lower()
#             food_words = food_lower.split()
#             food_word_count = len(food_words)
#             food_contexts[food] = [] # Store all food context for the food
            
#             # Search for the food name in the script
#             for i in range(len(words) - food_word_count + 1):
#                 potential_food = ' '.join(words[i:i+food_word_count])
                
#                 if potential_food == food_lower:
#                     # Extract context
#                     start = max(0, i - context_words)
#                     end = min(len(words), i + food_word_count + context_words)
#                     context = ' '.join(words[start:end])
#                     food_contexts[food].append(context)
        
#         return food_contexts
    
#     except Exception as e:
#         print(f"Error extracting contexts: {e}")
#         return {}

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
    script_path = find_script_file(movie_title, script_folder)
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