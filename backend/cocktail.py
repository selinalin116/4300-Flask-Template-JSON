import json
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
import math
import re
from helper_functions import tokenize_ingredients, get_cocktail_ingredients
import numpy as np

with open('data/cocktails.json', 'r') as f:
    cocktails = json.load(f)

cocktail_texts = []
cocktail_ingredients_list = [] 

for c in cocktails:
    ingredients = [str(c[f'strIngredient{i}']) for i in range(1, 16) if c.get(f'strIngredient{i}')]
    
    cocktail_ingredients_list.extend(ingredients)
    
    cocktail_text = " ".join([
        c['strDrink'],
        c['strInstructions'],
        " ".join(ingredients)
    ])
    
    cocktail_texts.append(cocktail_text)

cocktail_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
cocktail_tfidf = cocktail_vectorizer.fit_transform(cocktail_texts)

k = 40  # Same as class demo
U, s, vt = svds(cocktail_tfidf, k=k)
cocktail_vectors = normalize(U, axis=1)

def compute_idf():
    ing_list = [cocktail_ingredients_list]
    total_docs = len(cocktail_ingredients_list)
    ingredient_doc_freq = Counter()
    
    for ingredients in ing_list:
        unique_tokens = set()
        for ing in ingredients:
            tokens = tokenize_ingredients(ing)
            unique_tokens.update(tokens)
        ingredient_doc_freq.update(unique_tokens)

    idf = {
        ing: math.log((total_docs + 1) / (1 + ingredient_doc_freq[ing])) + 1
        for ing in ingredient_doc_freq
    }

    return idf

def jaccard_similarity(script, raw_ingredients):
    script_words = set(script.lower().split())
    ingredient_words = set(raw_ingredients)
    intersection = script_words & ingredient_words
    return len(intersection) / (len(script_words | ingredient_words) + 1e-8)

def get_semantic_profile(cocktail_index, vt, vectorizer, vectors, top_n=6):
    """
    Returns top-N semantic dimensions for a recipe as [{label, score}] pairs.
    Each label is the top word contributing to that SVD component.
    """
    vec = vectors[cocktail_index]
    top_indices = np.argsort(-vec)[:top_n]
    feature_names = vectorizer.get_feature_names_out()

    semantic_profile = []
    for i in top_indices:
        top_word_index = np.argmax(vt[i])
        label = feature_names[top_word_index]
        score = float(vec[i])
        semantic_profile.append({"label": label, "score": round(score, 4)})

    return semantic_profile

def clean_cocktail_data(cocktail, index=None, vt=None, vectorizer=None, vectors=None):

    semantic_profile = []
    if index is not None and vt is not None and vectorizer is not None and vectors is not None:
        semantic_profile = get_semantic_profile(index, vt, vectorizer, vectors)

    raw_ingredients, ingredients = get_cocktail_ingredients(cocktail)

    instructions = cocktail.get('strInstructions', '').strip()

    drink_id = cocktail.get('idDrink', '')
    drink_name = cocktail.get('strDrink', '').replace(" ", "-").lower()
    cocktail_url = f"https://www.thecocktaildb.com/drink/{drink_id}-{drink_name}"

    return {
        'name': cocktail.get('strDrink', 'Unnamed Cocktail').strip(),
        'image': cocktail.get('strDrinkThumb', '').strip(),
        'ingredients': ingredients,
        'raw_ingredients': raw_ingredients,
        'instructions': instructions,
        'recipe_link': cocktail_url,
        'semantic_profile': semantic_profile
    }

# def extract_ingredients(cocktail):
#     """
#     Extract all non-null strIngredient[x] values from a cocktail entry.
#     """
#     ingredients = []
#     for i in range(1, 16):
#         ingredient = cocktail.get(f'strIngredient{i}')
#         if ingredient:
#             ingredients.append(ingredient.strip())
#     return ingredients

def extract_ingredients(cocktail):
    """
    Extracts cleaned, non-null ingredients from a cocktail dictionary,
    removing modifier-only entries and stripping known descriptor words.
    """
    modifiers = {
        'fresh', 'chopped', 'diced', 'sliced', 'crushed', 'ground', 'minced',
        'large', 'small', 'medium', 'extra', 'shredded', 'grated', 'whole', 'cups',
        'plain', 'unsweetened', 'sweetened', 'semi-sweet', 'cooked', 'raw', 'all-purpose',
        'brown', 'heavy', 'unsalted', 'light', 'dark', 'hard', 'smoked',
        'half', 'green', 'hot', 'red', 'warm', 'lean', 'sour', 'food', 'sweet', 'mixed',
        'yellow', 'black', 'white', 'prepared', 'round', 'boiling', 'bay', 'dry', 'instant',
        'cut', 'dried', 'stuffed', 'live'
    }

    ingredients = []
    for i in range(1, 16):
        raw = cocktail.get(f'strIngredient{i}')
        if not raw:
            continue

        words = raw.lower().strip().split()
        # Remove modifiers
        cleaned_words = [w for w in words if w not in modifiers]

        if cleaned_words:  # Only keep ingredient if something remains
            cleaned_ingredient = ' '.join(cleaned_words)
            ingredients.append(cleaned_ingredient)

    return ingredients