import json
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
import math
import re
from helper_functions import tokenize_ingredients

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

# cocktail_ingredients = []
# for c in cocktails:
#     for i in range(1, 16):
#         if c.get(f'strIngredient{i}'):
#             cocktail_ingredients.append(str(c[f'strIngredient{i}']))

cocktail_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
cocktail_tfidf = cocktail_vectorizer.fit_transform(cocktail_texts)

# cocktail_ingredients_vectorizer = TfidfVectorizer(
#     stop_words='english', 
#     max_df=0.9,
#     min_df=1
# )
# cocktail_ingredients_tfidf = cocktail_ingredients_vectorizer.fit_transform(cocktail_ingredients)

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

def clean_cocktail_data(cocktail):
    ingredients = []
    raw_ingredients = []
    for i in range(1, 16):
        ingredient = cocktail.get(f'strIngredient{i}')
        measure = cocktail.get(f'strMeasure{i}', '') or ''
        measure = str(measure).strip()

        if ingredient and ingredient.strip():
            raw_ingredients.append(ingredient.strip().lower())
            ingredients.append(
                f"{ingredient.strip()} ({measure})" if measure 
                else ingredient.strip()
            )

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
        'recipe_link': cocktail_url
    }

def extract_ingredients(cocktail):
    """
    Extract all non-null strIngredient[x] values from a cocktail entry.
    """
    ingredients = []
    for i in range(1, 16):
        ingredient = cocktail.get(f'strIngredient{i}')
        if ingredient:
            ingredients.append(ingredient.strip())
    return ingredients
