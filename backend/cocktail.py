import requests
import json
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import os
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import os

ingredient_drink_index = {}

ingredient_url = "https://www.thecocktaildb.com/api/json/v1/1/list.php?i=list"
ingredient_response = requests.get(ingredient_url)
ingredients = ingredient_response.json().get("drinks", [])

# format of the ingredient_url website
# "drinks":[{"strIngredient1":"Light rum"},{"strIngredient1":"Bourbon"},
# {"strIngredient1":"Vodka"},{"strIngredient1":"Gin"},{"strIngredient1":"Blended whiskey"},{"strIngredient1":"Tequila"}
for x in ingredients:
    ingredient = x["strIngredient1"]
    drinks_url = f"https://www.thecocktaildb.com/api/json/v1/1/filter.php?i={ingredient}"
    # here's what we get w/ ingredient = vodka
    # "drinks":[{"strDrink":"155 Belmont","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/yqvvqs1475667388.jpg",
    # "idDrink":"15346"},{"strDrink":"501 Blue","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/ywxwqs1461867097.jpg",
    # "idDrink":"17105"},{"strDrink":"57 Chevy with a White License 

    drinks_response = requests.get(drinks_url)
    # get drink names
    drink_list = []
    for y in drinks_response.json().get("drinks", []):
        drink = y["strDrink"]
        drink_list.append(drink)
    ingredient_drink_index[ingredient] = drink_list

# write to file
with open("ingredient_drink_index.json", "w") as f:
    json.dump(ingredient_drink_index, f, indent=4)


def fetch_cocktails():
    cocktails = []
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        response = requests.get(f'https://www.thecocktaildb.com/api/json/v1/1/search.php?f={letter}')
        if response.ok and response.json().get('drinks'):
            cocktails.extend(response.json()['drinks'])
    return cocktails

cocktails = fetch_cocktails()
print(cocktails)

cocktail_texts = [
    " ".join([
        c['strDrink'], 
        c['strInstructions'],
        " ".join(str(c[f'strIngredient{i}']) for i in range(1, 16) if c.get(f'strIngredient{i}'))
    ])
    for c in cocktails
]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
cocktail_tfidf = vectorizer.fit_transform(cocktail_texts)

k = 40  # Same as class demo
U, s, vt = svds(cocktail_tfidf, k=k)
cocktail_vectors = normalize(U, axis=1)

def get_movie_script(movie_title, folder):
    """Your existing script loader from P02"""
    script_path = os.path.join(folder, f"{movie_title.lower().replace(' ', '_')}.txt")
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            return f.read()
    return None

def jaccard_similarity(script, ingredients):
    script_words = set(script.lower().split())
    ingredient_words = set(ingredients.lower().split())
    intersection = script_words & ingredient_words
    return len(intersection) / (len(script_words | ingredient_words) + 1e-8)

def clean_cocktail_data(cocktail):
    """Extract only name, ingredients, and image from cocktail data"""
    ingredients = []
    for i in range(1, 16):
        ingredient = cocktail.get(f'strIngredient{i}')
        measure = cocktail.get(f'strMeasure{i}', '') or ''
        measure = str(measure).strip()
        
        if ingredient:
            ingredients.append(
                f"{ingredient} ({measure})" if measure 
                else ingredient
            )
    
    return {
        'name': cocktail.get('strDrink', 'Unnamed Cocktail'),
        'image': cocktail.get('strDrinkThumb', ''),
        'ingredients': ingredients
    }
