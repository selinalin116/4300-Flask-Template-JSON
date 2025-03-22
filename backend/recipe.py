from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import pandas as pd
import json
import zipfile

zip_filename = 'data/recipes_cleaned.zip'
file_inside_zip = 'recipes_cleaned.json'

# Open the ZIP file and read the file inside it
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    with zipf.open(file_inside_zip) as file:
        data = file.read().decode('utf-8')

recipes = json.loads(data)

with open("data/recipes_cleaned.json") as file:
    recipes = json.load(file)

recipe_descriptions = [description for description in recipes['description']]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
recipe_tfidf = vectorizer.fit_transform(recipe_descriptions)

k = 40  # Same as class demo
U, s, vt = svds(recipe_tfidf, k=k)
recipe_vectors = normalize(U, axis=1)