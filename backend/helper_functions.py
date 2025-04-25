import os
import re
import string
import nltk
import ssl
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import ast
import Levenshtein

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

    # Remove parenthetical directions
    text = re.sub(r'\(.*?\)', '', text)  

    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))

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
                break

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

def generate_ngrams(tokens, n):
    """
    Generates a list of n-grams (contiguous sequences of n tokens) from a list of tokens
    """
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def get_script_phrases(script_text, min_word_length=3):
    """
    Tokenize and return a set of unigrams, bigrams, and trigrams from the script.
    """
    tokens = tokenize_script(script_text, min_word_length)
    phrases = set(tokens)  # unigrams

    for n in [2, 3]:  # Add bigrams and trigrams
        phrases.update(generate_ngrams(tokens, n))

    return phrases

# def normalize_ingredient(ingredient):
#     """
#     Normalize ingredient phrase to remove common descriptors
#     """
#     phrase = ingredient.lower().strip()
#     results = set()

#     # Compounds where we don't want to extract the base ingredient since the individual words do not have much significant meanings
#     no_base_extraction = ['baking powder', 'baking soda']
    
#     # Common compound ingredints
#     compound_ingredients = [
#         'olive oil', 'vegetable oil', 'peanut butter', 'cream cheese', 
#         'sour cream', 'maple syrup',
#         'coconut milk', 'heavy cream', 'red wine', 'white wine', 
#     ]
    
#     all_compounds = no_base_extraction + compound_ingredients

#     for compound in all_compounds:
#         if compound in phrase:
#             results.add(compound)
            
#             if compound not in no_base_extraction:
#                 base_ingredient = compound.split()[-1]  # Gets the last word of a compound ingredient since that is usually a noun
#                 if base_ingredient not in results:
#                     results.add(base_ingredient)
#             else:
#                 return

#     phrase = ingredient.lower().strip()
    
#     # Common descriptors in ingredients
#     modifiers = [
#         'fresh', 'chopped', 'diced', 'sliced', 'crushed', 'ground', 'minced',
#         'large', 'small', 'medium', 'extra', 'shredded', 'grated', 'whole', 'cups',
#         'plain', 'unsweetened', 'sweetened', 'semi-sweet', 'cooked', 'raw', 'all-purpose',
#         'brown', 'heavy', 'unsalted', 'light', 'dark', 'hard', 'smoked',
#         'half', 'green', 'hot', 'red', 'warm', 'lean', 'sour', 'food', 'sweet', 'mixed', 'yellow', 'black', 'white', 'prepared', 'round', 'boiling', 'bay', 'dry', 'instant', 'cut', 'dried', 'fresh', 'stuffed'
#     ]

#     # Remove descriptors
#     words = phrase.split()
#     filtered_words = [word for word in words if word not in modifiers]

#     if filtered_words:
#         results.add(" ".join(filtered_words))  # cleaned full phrase
#         results.update(filtered_words)   
    
#     return list(results)

def normalize_ingredient(ingredient):
    """
    Normalize ingredient phrase to remove common descriptors
    """
    phrase = ingredient.lower()
    results = []

    # Compounds where we don't want to extract the base ingredient since the individual words do not have much significant meanings
    no_base_extraction = ['baking powder', 'baking soda']
    
    # Common compound ingredints
    compound_ingredients = [
        'olive oil', 'vegetable oil', 'peanut butter', 'cream cheese', 
        'sour cream', 'maple syrup',
        'coconut milk', 'heavy cream', 'red wine', 'white wine', 
    ]
    
    all_compounds = no_base_extraction + compound_ingredients

    for compound in all_compounds:
        if compound in phrase:
            results.append(compound)
            
            if compound not in no_base_extraction:
                base_ingredient = compound.split()[-1]  # Gets the last word of a compound ingredient since that is usually a noun
                if base_ingredient not in results:
                    results.append(base_ingredient)
            else:
                return

    phrase = ingredient.lower().strip()
    
    # Common descriptors in ingredients
    modifiers = [
        'fresh', 'chopped', 'diced', 'sliced', 'crushed', 'ground', 'minced',
        'large', 'small', 'medium', 'extra', 'shredded', 'grated', 'whole', 'cups',
        'plain', 'unsweetened', 'sweetened', 'semi-sweet', 'cooked', 'raw', 'all-purpose',
        'brown', 'heavy', 'unsalted', 'light', 'dark', 'hard', 'smoked',
        'half', 'green', 'hot', 'red', 'warm', 'lean', 'sour', 'food', 'sweet', 'mixed', 'yellow', 'black', 'white', 'prepared', 'round', 'boiling', 'bay', 'dry', 'instant', 'cut', 'dried', 'fresh', 'stuffed', 'live'
    ]

    # Remove descriptors
    words = phrase.split()
    filtered_words = [word for word in words if word not in modifiers]

    normalized = ' '.join(filtered_words).strip()
    if normalized and normalized not in results:
        results.append(normalized)
    
    if len(filtered_words) >= 2:
        last_one = filtered_words[-1]
        second_last = filtered_words[-2]
        if second_last not in results:
            results.append(second_last)
        if last_one not in results:
            results.append(last_one)
    elif filtered_words:
        if filtered_words[0] not in results:
            results.append(filtered_words[0])
    
    return results if results else None


COMMON_INGREDIENTS = {"water", "salt", "sugar", "hot", "damn"}

def build_combined_weight_dict(script_words, idf_lookup, boost=1.5, common_penalty=0.2):
    """
    Build a weight dictionary based on a idf lookup table and scripts
    """
    final_weights = {}

    for word in idf_lookup:
        idf_weight = idf_lookup.get(word, 1.0)
        boost = boost if word in script_words else 1.0
        penalty = common_penalty if word in COMMON_INGREDIENTS else 1.0

        final_weight = idf_weight * boost * penalty
        final_weights[word] = final_weight

    return final_weights

def tokenize_ingredients(ingredient):
    """
    Tokenize an individual ingredients
    """
    return re.findall(r'\b[a-zA-Z]+\b', ingredient.lower())

def weighted_jaccard_similarity(script_words, ingredient_set, weight_dict):
    """
    A version of jaccard similarity that takes adds more weights for ingredients words that appears in scripts
    """
    intersection = script_words & ingredient_set

    # Focus on the percentage of ingredients shared
    intersection_weight = sum(weight_dict.get(word, 1.0) for word in intersection)
    ingredient_weight = sum(weight_dict.get(word, 1.0) for word in ingredient_set)

    return intersection_weight / ingredient_weight if ingredient_weight != 0 else 0.0

def idf_jaccard_similarity(script_words, ingredient_set, weight_dict, raw_weight_dict):
    """
    A version of jaccard similarity that takes idf into account
    """
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
    """
    A version of jaccard similarity that takes idf and ingredient list length into account
    """

    intersection = script_words & ingredient_set
    union = script_words | ingredient_set

    intersection_weight = sum(weight_dict.get(word, 1.0) for word in intersection)
    ingredient_weight = sum(weight_dict.get(word, 1.0) for word in ingredient_set)

    raw_intersection_weight = sum(raw_weight_dict.get(word, 1.0) for word in intersection)
    raw_ingredient_weight = sum(raw_weight_dict.get(word, 1.0) for word in ingredient_set)
    
    weighted_score = intersection_weight / ingredient_weight if ingredient_weight != 0 else 0.0
    raw_jaccard = raw_intersection_weight / raw_ingredient_weight if union else 0.0

    # Penalize short ingredient lists
    penalty = len(ingredient_set) / (len(ingredient_set) + 3) 
    penalized_score = weighted_score * penalty

    return penalized_score, raw_jaccard

def cosine_sim(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return sim

def combine_scores(jaccard_score, svd_score, alpha = .25):
    """
    Combines Jaccard and SVD similarity scores using a weighted average.
    """
    return alpha * jaccard_score + (1 - alpha) * svd_score


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

def is_edit_distance_match(word, target, max_dist=2):
    """
    Edit distance function to look for similar words
    """
    return Levenshtein.distance(word.lower(), target.lower()) <= max_dist

def edit_normalize_ingredient(ingredient):
    """
    Normalizes an ingredient string by matching it to a synonym key 
    using edit distance.
    """
    for key in synonym_map:
        if is_edit_distance_match(ingredient, key):
            return key
    return ingredient

synonym_map = {
    "homecooked": ["homemade", "comfort", "family", "cooked", "traditional"]
}

def embed_ingredient_list(ingredients, model):
    """
    Takes a list of ingredients like ["lime juice", "simple syrup"]
    and returns the averaged embedding vector using pretrained model.
    """
    all_vectors = []

    for ingredient in ingredients:
        normalized = edit_normalize_ingredient(ingredient)
        words = normalized.lower().split()

        if normalized in synonym_map:
            words += synonym_map[normalized]

        for word in words:
            if word in model:
                all_vectors.append(model[word])

    if not all_vectors:
        return None
    return sum(all_vectors) / len(all_vectors)

def get_cocktail_ingredients(cocktail):
    """
    Returns the ingredients of a cocktail
    """
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
    
    return raw_ingredients, ingredients

meat_no_fish = ['chicken','bacon','turkey','beef','pork','duck','steak','wings',
                'boneless skinless chicken breast halves','ham','veal','lamb', 'sausage',
                'ground chuck','suet','ox kidney']
fish = ['fish', 'salmon', 'sardines','trout', 'mackerel', 'cod', 'haddock', 'pollock',
        'flounder', 'tilapia', 'shellfish', 'mussels', 'scallops', 'squid', 
        'oysters', 'crab', 'shrimp', 'sea bass', 'halibut', 'tuna','clams',
        'lobster','anchovy','marlin steaks','conch','caviar']
dairy = ['milk', 'ice cream', 'cheese', 'yoghurt', 'yogurt', 'cream', 'butter', 
            'buttermilk', 'heavy cream', 'butter', 'egg','custard',
            'half-and-half','marscarpone','eggs','heavy whipping cream', 'chocolate']
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
def dietary_res(items, top_k=6, restrictions=None):
    """
    Helper to filter and return top_k items based on dietary restrictions
    """

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
def ingredients_to_drink(drink, pref):
    filtered = []
    for item in drink:
        ingredients = []
        for i in range(1, 16):
            ingredients += [item[0][(f"strIngredient{i}")]]
        violates = False
        for restriction in pref:
            restricted_ings = restriction_map.get(restriction, [])
            if any(restricted in ing.lower() if ing else '' for ing in ingredients for restricted in restricted_ings):
                violates = True
                break

        if not violates:
            filtered.append(item)

    return filtered

def extract_alcohol_phrases(word_list):
    phrases = []
    i = 0
    while i < len(word_list) - 1:
        pair = f"{word_list[i].lower()} {word_list[i+1].lower()}"
        if pair in {
            "non alcoholic",
            "no alcohol",
            "alcohol only",
            "only alcohol"
        }:
            if pair=="non alcoholic" or pair=="no alcohol":
                phrases.append("non alcoholic")
            else:
                phrases.append("alcoholic")
            i += 2
        else:
            i += 1
    return phrases