import os
import re
import string
import nltk
import ssl
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import ast
# Fix for SSL certificate verification issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import ngrams

def capitalize_sentences(text):
    sentences = sent_tokenize(text)
    capitalized = [s.strip().capitalize() for s in sentences]
    return " ".join(capitalized)

def tokenize_script(script_text, min_word_length=3):
    """
    Tokenize the script, remove stopwords and punctuation,
    and return a list of meaningful terms.
    """
    if not script_text:
        return []
    
    text = script_text.lower()

    # Remove script formatting like scene headings, character names
    text = re.sub(r'\b(INT|EXT|FADE IN|FADE OUT|CUT TO|DISSOLVE TO)\..*', '', text)
    text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical directions

    # Tokenize
    tokens = word_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))

    # Add common script terms
    script_specific_stopwords = {
        'scene', 'cut', 'fade', 'dissolve', 'angle', 'shot', 'ext', 'int',
        'interior', 'exterior', 'day', 'night', 'continued', 'vo', 'os'
    }
    stop_words.update(script_specific_stopwords)

    lemmatizer = WordNetLemmatizer()

    # Filter out stopwords, punctuation, and short words
    filtered_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in stop_words
        and word not in string.punctuation
        and len(word) >= min_word_length
        and word.isalpha()
    ]
    
    return filtered_tokens

def get_movie_script(movie_title, folder, min_word_length=3):
    """
    Load and return the content of a movie script.
    Supports exact and partial matching with hyphens or underscores.
    """
    base = movie_title.lower()
    variations = [
        f"{base.replace(' ', '-')}.txt",
        f"{base.replace(' ', '_')}.txt"
    ]
    
    script_content = None

    # Try exact filename matches first
    for filename in variations:
        script_path = os.path.join(folder, filename)
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()

    # Fallback: try partial match
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            filename_lower = file.lower()
            # Check if any word from input appears in filename
            input_words = base.split()
            if all(word in filename_lower for word in input_words):
                script_path = os.path.join(folder, file)
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = f.read()
                    break
    
    # If no script found, return None
    if script_content is None:
        return None
    
    tokens = tokenize_script(script_content, min_word_length)

    return ' '.join(tokens)

COMMON_INGREDIENTS = {"water", "salt", "sugar", "hot", "damn"}

def build_combined_weight_dict(script_words, idf_lookup, boost=1.5, common_penalty=0.2):
    final_weights = {}

    for word in idf_lookup:
        idf_weight = idf_lookup.get(word, 1.0)
        boost = boost if word in script_words else 1.0
        penalty = common_penalty if word in COMMON_INGREDIENTS else 1.0

        final_weight = idf_weight * boost * penalty
        final_weights[word] = final_weight

        print(f"{word}: {final_weight:.4f}")
    return final_weights

def tokenize_ingredients(ingredient):
    return re.findall(r'\b[a-zA-Z]+\b', ingredient.lower())

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def weighted_jaccard_similarity(script_words, ingredient_set, weight_dict):
    intersection = script_words & ingredient_set

    # Focus on the percentage of ingredients shared
    intersection_weight = sum(weight_dict.get(word, 1.0) for word in intersection)
    ingredient_weight = sum(weight_dict.get(word, 1.0) for word in ingredient_set)

    return intersection_weight / ingredient_weight if ingredient_weight != 0 else 0.0

def idf_jaccard_similarity(script_words, ingredient_set, weight_dict, raw_weight_dict):
    intersection = script_words & ingredient_set
    union = script_words | ingredient_set

    # Focus on the percentage of ingredients shared
    intersection_weight = sum(weight_dict.get(word, 1.0) for word in intersection)
    ingredient_weight = sum(weight_dict.get(word, 1.0) for word in ingredient_set)

    raw_intersection_weight = sum(raw_weight_dict.get(word, 1.0) for word in intersection)
    raw_ingredient_weight = sum(raw_weight_dict.get(word, 1.0) for word in ingredient_set)

    weighted_score = intersection_weight / ingredient_weight if ingredient_weight != 0 else 0.0
    raw_jaccard = raw_intersection_weight / raw_ingredient_weight if union else 0.0

    return weighted_score, raw_jaccard

def penalize_jaccard_similarity(script_words, ingredient_set, weight_dict, raw_weight_dict):
    intersection = script_words & ingredient_set
    union = script_words | ingredient_set

    intersection_weight = sum(weight_dict.get(word, 1.0) for word in intersection)
    ingredient_weight = sum(weight_dict.get(word, 1.0) for word in ingredient_set)

    raw_intersection_weight = sum(raw_weight_dict.get(word, 1.0) for word in intersection)
    raw_ingredient_weight = sum(raw_weight_dict.get(word, 1.0) for word in ingredient_set)
    
    weighted_score = intersection_weight / ingredient_weight if ingredient_weight != 0 else 0.0
    raw_jaccard = raw_intersection_weight / raw_ingredient_weight if union else 0.0

    # Penalize short ingredient lists
    penalty = len(ingredient_set) / (len(ingredient_set) + 2) 
    penalized_score = weighted_score * penalty

    return penalized_score, raw_jaccard

# def weighted_jaccard(set1, set2, idf_dict):
#     intersection = set1.intersection(set2)
#     union = set1.union(set2)

#     intersection_weight = sum(idf_dict.get(item, 0.0) for item in intersection)
#     union_weight = sum(idf_dict.get(item, 0.0) for item in union)

#     if union_weight == 0:
#         return 0.0
#     return intersection_weight / union_weight

def combine_scores(jaccard_score, svd_score, alpha = .25):
    return alpha * jaccard_score + (1 - alpha) * svd_score

# def cosine_similarity(script, recipes, recipe_vectorizer):
#     """
#     Calculate cosine similarity between script and recipes using existing TF-IDF vectorizer
#     """
#     # Transform script using existing recipe vectorizer
#     script_tfidf = recipe_vectorizer.transform([script])
    
#     # Get recipe ingredients as text for vectorization
#     recipe_texts = [" ".join(recipe['ingredients']) for recipe in recipes]
    
#     # Transform recipes using the same vectorizer
#     recipe_tfidf = recipe_vectorizer.transform(recipe_texts)
    
#     # Calculate cosine similarity directly between script and each recipe
#     similarities = sklearn_cosine_similarity(script_tfidf, recipe_tfidf)[0]
    
#     return similarities

def cosine_similarity(ingredients_tfidf, description, vectorizer):
    description_tfidf = vectorizer.transform([description])
    
    # Calculate similarity (0-1 score)
    similarity = sklearn_cosine_similarity(ingredients_tfidf, description_tfidf)[0]
    
    # Convert to percentage
    return similarity


def description_svd(vectorizer, additional_description, vt, vectors):
    """
    Calculate similarity scores between an additional text description and a set of vectors using SVD
    """
    desc_similarities = None
    if additional_description:
        additional_tfidf = vectorizer.transform([additional_description])
        additional_projected = additional_tfidf.dot(vt.T)
        desc_similarities = additional_projected.dot(vectors.T)

    return desc_similarities

def embed_ingredient_list(ingredients, model):
    """
    Takes a list of ingredients like ["lime juice", "simple syrup"]
    and returns the averaged embedding vector using pretrained model.
    """
    all_vectors = []
    
    for ingredient in ingredients:
        words = ingredient.lower().split() 
        for word in words:
            if word in model:
                all_vectors.append(model[word])
    
    if not all_vectors:
        return None
    return sum(all_vectors) / len(all_vectors)


def dietary_res(items, top_k=6, restrictions=None):
    """
    Helper to filter and return top_k items based on dietary restrictions.

    Parameters:
    - items: list of food items
    - top_k: number of top items to return.
    - restrictions: list of restricted diets (ie. ['vegetarian', 'lactose'])

    Returns:
    - Filtered and sorted list of top_k items.
    """
    meat_no_fish = ['chicken','bacon','turkey','beef','pork','duck','steak','wings',
                    'boneless skinless chicken breast halves','ham','veal','lamb', 'sausage',
                    'ground chuck','suet','ox kidney']
    fish = ['fish', 'salmon', 'sardines','trout', 'mackerel', 'cod', 'haddock', 'pollock',
            'flounder', 'tilapia', 'shellfish', 'mussels', 'scallops', 'squid', 
            'oysters', 'crab', 'shrimp', 'sea bass', 'halibut', 'tuna','clams',
            'lobster','anchovy','marlin steaks','conch','caviar']
    dairy = ['milk', 'ice cream', 'cheese', 'yogurt', 'cream', 'butter', 
             'buttermilk', 'heavy cream', 'butter', 'egg','custard',
             'half-and-half','marscarpone','eggs','heavy whipping cream']
    gluten_food = ['bread', 'beer', 'cake', 'pie', 'candy', 'cereal', 'cookie', 'croutons', 'french fries',
                   'gravy', 'seafood', 'malt', 'pasta', 'hot dog', 'salad dressing', 'soy sauce', 'rice seasoning', 
                   'chips', 'chicken', 'soup','flour','wheat','pastry','couscous','semolina','bulgar','barley','rye','oats',
                   'spelt','deitan','graham crackers','pretzel']
    non_kosher = ['shellfish', 'crab', 'shrimp', 'lobster', 'pork']

    restriction_map = {
        "vegan": meat_no_fish + fish + dairy,
        "vegetarian": meat_no_fish + fish,
        "pescatarian": meat_no_fish,
        "dairy-free": dairy,
        "gluten-free": gluten_food,
        "kosher": non_kosher,
    }

    filtered = []
    for item in items:
        ingredients = item[0]["ingredients"]
        ingredients = ast.literal_eval(ingredients)
        violates = False
        for restriction in restrictions:
            restricted_ings = restriction_map.get(restriction, [])
            if any(restricted in ing.lower() for ing in ingredients for restricted in restricted_ings):
                violates = True
                break

        if not violates:
            filtered.append(item)

        if len(filtered) >= top_k:
            break

    return filtered


def drinks_filtered(items, top_k, preferences):
    """
    Helper to filter and return top_k items for user's drink preferences.

    Parameters:
    - items: list of drink items
    - top_k: number of top items to return.
    - restrictions: list of preference (ie. ['alcoholic'])

    Returns:
    - Filtered and sorted list of top_k items.
    """
    filtered = []
    for item in items:
        alc = [item[0]["strAlcoholic"].lower()]
        # print("alc",alc)
        # print("pref",preferences)
        if (alc==preferences):
            filtered.append(item)

        if len(filtered) >= top_k:
            break

    return filtered