import json
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ingredient_drink_index = {}

# ingredient_url = "https://www.thecocktaildb.com/api/json/v1/1/list.php?i=list"
# ingredient_response = requests.get(ingredient_url)
# ingredients = ingredient_response.json().get("drinks", [])

# format of the ingredient_url website
# "drinks":[{"strIngredient1":"Light rum"},{"strIngredient1":"Bourbon"},
# {"strIngredient1":"Vodka"},{"strIngredient1":"Gin"},{"strIngredient1":"Blended whiskey"},{"strIngredient1":"Tequila"}
# for x in ingredients:
#     ingredient = x["strIngredient1"]
#     drinks_url = f"https://www.thecocktaildb.com/api/json/v1/1/filter.php?i={ingredient}"
#     # here's what we get w/ ingredient = vodka
#     # "drinks":[{"strDrink":"155 Belmont","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/yqvvqs1475667388.jpg",
#     # "idDrink":"15346"},{"strDrink":"501 Blue","strDrinkThumb":"https:\/\/www.thecocktaildb.com\/images\/media\/drink\/ywxwqs1461867097.jpg",
#     # "idDrink":"17105"},{"strDrink":"57 Chevy with a White License 

#     drinks_response = requests.get(drinks_url)
#     # get drink names
#     drink_list = []
#     for y in drinks_response.json().get("drinks", []):
#         drink = y["strDrink"]
#         drink_list.append(drink)
#     ingredient_drink_index[ingredient] = drink_list

# # write to file
# with open("ingredient_drink_index.json", "w") as f:
#     json.dump(ingredient_drink_index, f, indent=4)

# def fetch_cocktails():
#     """
#     Fetch cocktails from cocktaildb API.
#     """
#     cocktails = []
#     for letter in 'abcdefghijklmnopqrstuvwxyz':
#         response = requests.get(f'https://www.thecocktaildb.com/api/json/v1/1/search.php?f={letter}')
#         if response.ok and response.json().get('drinks'):
#             cocktails.extend(response.json()['drinks'])
#     return cocktails

# cocktails = fetch_cocktails()

with open('data/cocktails.json', 'r') as f:
    cocktails = json.load(f)

# cocktail_texts = [
#     " ".join([
#         c['strDrink'], 
#         c['strInstructions'],
#         " ".join(str(c[f'strIngredient{i}']) for i in range(1, 16) if c.get(f'strIngredient{i}'))
#     ])
#     for c in cocktails
# ]

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
