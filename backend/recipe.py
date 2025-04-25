from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import json
import ast
import helper_functions
import math
from collections import Counter
import ssl
import nltk
from textblob import TextBlob

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('brown')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('movie_reviews')

with open('data/recipes_cleaned.json', 'r') as f:
    recipes = json.load(f)

import ast

def parse_steps_to_paragraph(steps_string):
    """
    Converts a string representation of a list of steps into a readable paragraph.
    """
    try:
        if steps_string.startswith('"') and steps_string.endswith('"'):
            steps_string = steps_string[1:-1]

        steps_list = ast.literal_eval(steps_string)

        cleaned_steps = []
        for step in steps_list:
            step = step.strip().capitalize()
            if not step.endswith('.'):
                step += '.'
            cleaned_steps.append(step)

        paragraph = " ".join(cleaned_steps)
        return paragraph

    except (SyntaxError, ValueError):
        # Fallback in case of bad formatting
        cleaned = steps_string.strip("[]'\"")
        rough_steps = [step.strip(" '\"") for step in cleaned.split("', '")]

        cleaned_steps = []
        for step in rough_steps:
            step = step.strip().capitalize()
            if not step.endswith('.'):
                step += '.'
            cleaned_steps.append(step)

        return " ".join(cleaned_steps)

    
recipe_instructions = [parse_steps_to_paragraph(i['steps']) for i in recipes]

recipe_descriptions = [i['description'] for i in recipes]

recipe_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
recipe_tfidf = recipe_vectorizer.fit_transform(recipe_descriptions)

k = 40  # recipesvd has a graph that explains this choice
u, s, rec_vt = svds(recipe_tfidf, k=k)
recipe_vectors = normalize(u, axis=1)

combined_texts = [
    f"{desc} {instr}"
    for desc, instr in zip(recipe_descriptions, recipe_instructions)
]

recipe_vectorizer_instructions = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
recipe_tfidf_instructions = recipe_vectorizer_instructions.fit_transform(combined_texts)

i_u, i_s, i_rec_vt = svds(recipe_tfidf_instructions, k=k)
i_recipe_vectors = normalize(i_u, axis=1)

recipe_ingredient_lists = []

for r in recipes:
    try:
        ingredients_list = ast.literal_eval(r['ingredients'])
        tokens = set()
        for ing in ingredients_list:
            tokens.update(helper_functions.tokenize_ingredients(ing))
        recipe_ingredient_lists.append(list(tokens)) 
    except (ValueError, SyntaxError):
        continue

def recipe_compute_idf():
    total_docs = len(recipe_ingredient_lists)
    doc_freq = Counter()

    for ingredients in recipe_ingredient_lists:
        unique_tokens = set(ingredients)
        doc_freq.update(unique_tokens)

    idf_lookup = {
        token: math.log((total_docs + 1) / (1 + doc_freq[token])) + 1
        for token in doc_freq
    }

    return idf_lookup

def get_sentiment(reviews):
    if reviews is None:
        return None
    clean_reviews = [str(r) for r in reviews if isinstance(r, str) and r.strip()]

    if not clean_reviews:
        return None
    
    combined = ' '.join(clean_reviews)
    return TextBlob(combined).sentiment.polarity

def get_sentiment_text(score):
    if score is None: 
        return "No Ratings"
    elif score < -0.5:
        return "Very Negative"
    elif -0.5 <= score < -0.1:
        return "Fairly Negative"
    elif -0.1 <= score <= 0.1:
        return "Neutral"
    elif 0.1 < score <= 0.5:
        return "Fairly Positive"
    else:
        return "Very Positive"

def clean_recipe_data(recipe):
    """
    Extracts and formats key details from a recipe dictionary.
    """
    instructions = parse_steps_to_paragraph(recipe["steps"])

    return {
        'name': recipe['name'].title(),
        'description': helper_functions.capitalize_sentences(recipe['description']),
        'ingredients': [ingredient.strip().lower() for ingredient in ast.literal_eval(recipe['ingredients'])],
        'rating': recipe['average_rating'],
        'rating_count': recipe['review_count'],
        'instructions': instructions 
    }