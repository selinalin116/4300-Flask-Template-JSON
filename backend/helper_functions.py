import os
import re
import string
import nltk
import ssl
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

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


def jaccard_similarity(set1, set2):
    """
    Calculate jaccard similarity between two sets.
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def weighted_jaccard(set1, set2, idf_dict):
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    intersection_weight = sum(idf_dict.get(item, 0.0) for item in intersection)
    union_weight = sum(idf_dict.get(item, 0.0) for item in union)

    if union_weight == 0:
        return 0.0
    return intersection_weight / union_weight

def combine_scores(jaccard_score, svd_score, alpha = .5):
    """
    Combines two similarity scores (Jaccard and SVD-based) using a weighted sum.
    """
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