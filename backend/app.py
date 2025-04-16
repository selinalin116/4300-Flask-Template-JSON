from cocktail import *
from recipe import recipe_vectors, recipes, clean_recipe_data, rec_vt, recipe_vectorizer, i_rec_vt, i_recipe_vectors, recipe_vectorizer_instructions
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import helper_functions
import ast
from cocktail import extract_ingredients
from helper_functions import weighted_jaccard_similarity
from gensim.models import KeyedVectors
import numpy as np

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

SCRIPT_FOLDER = os.path.join(current_directory, 'data/scripts')  
# FOOD_DATABASE = os.path.join(current_directory, 'data/recipes_names.csv')  
FOOD_DATABASE = os.path.join(current_directory, 'database.txt')  
MOVIE_DATABASE = os.path.join(current_directory, 'data/movies.txt')  # A file containing a list of movie titles

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template('base.html', title="Movie Food Finder")

@app.route("/find-foods")
def find_foods():
    """
    API endpoint to find foods in a movie script
    """
    model = model = KeyedVectors.load("model/glove.kv")  # Loads way faster

    movie_title = request.args.get('movie', '').strip()
    
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"})
    
    drink_description = None
    food_description = None
    if 'drinkdescription' in request.args:
        drink_description = request.args.get('drinkdescription').strip()

    if 'fooddescription' in request.args:
        food_description = request.args.get('fooddescription').strip()
    
    script = helper_functions.get_movie_script(movie_title, SCRIPT_FOLDER)
    if not script:
        return jsonify({"error": "Script not found"})
    
    script_tfidf = cocktail_vectorizer.transform([script])
    script_projected = script_tfidf.dot(vt.T)
    script_projected = normalize(script_projected)
    
    script_words = set(script.split())

    # Extract cocktail ingredients
    cocktail_ingredients_set = set()
    for cocktail in cocktails:
        cocktail_ingredients = extract_ingredients(cocktail)
        cocktail_ingredients_set.update(ingredient.lower() for ingredient in cocktail_ingredients)

    # Extract recipe ingredients
    recipe_ingredients_set = set()
    for recipe in recipes:
        try:
            ingredients_list = ast.literal_eval(recipe['ingredients'])
            recipe_ingredients_set.update(ing.lower().strip() for ing in ingredients_list)
        except (SyntaxError, ValueError):
            pass

    similarities = script_projected.dot(cocktail_vectors.T)

    beta = 0.9

    # cocktail_desc_similarities = helper_functions.description_svd(cocktail_vectorizer, drink_description, vt, cocktail_vectors)
    
    weight_dict = {word: 1.5 for word in script_words}

    cocktail_jaccard_scores = []
    cocktail_cosine_scores = []
    for cocktail in cocktails:
        cocktail_ingredients = extract_ingredients(cocktail)
        cocktail_ingredients_set = set(" ".join(cocktail_ingredients).lower().split())
        jaccard_score = weighted_jaccard_similarity(script_words, cocktail_ingredients_set, weight_dict)
        cocktail_jaccard_scores.append(jaccard_score)
        # cosine_score = helper_functions.cosine_similarity(cocktail_ingredients_tfidf, drink_description, cocktail_ingredients_vectorizer)
        # print(cosine_score)

        if drink_description is not None:
            query_vec = helper_functions.embed_ingredient_list([drink_description], model)
            cocktail_vec = helper_functions.embed_ingredient_list(cocktail_ingredients, model)

            if query_vec is not None and cocktail_vec is not None:
                sim = np.dot(query_vec, cocktail_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(cocktail_vec))
                cocktail_cosine_scores.append(sim)
                # print("Similarity:", sim)
            else:
                print("Could not compute similarity")

    
    # combine jaccard similarity for cocktails
    combined_cocktail_scores = []
    for i, cocktail in enumerate(cocktails):
        svd_text_score = similarities[0][i]
        cosine_score = None
        if drink_description is not None:
            # svd_desc_score = cocktail_desc_similarities[0][i]
            cosine_score = cocktail_cosine_scores[i]
            combined_svd_score = (1 - beta) * svd_text_score + beta * cosine_score
        else:
            combined_svd_score = svd_text_score

        jaccard_score = cocktail_jaccard_scores[i]
        combined_score = helper_functions.combine_scores(jaccard_score, combined_svd_score, alpha=0.5)
        combined_cocktail_scores.append((cocktail, combined_score, jaccard_score, svd_text_score, cosine_score))

    combined_cocktail_scores = sorted(combined_cocktail_scores, key=lambda x: -x[1])

    # Sort and Get Top Cocktails
    top_cocktails = [
    {
        "data": clean_cocktail_data(cocktail),
        "score": round(score * 100, 1),
        "jaccard_score": round(jaccard_score * 100, 1),
        "svd_text_score": round(svd_text_score * 100, 1),
        "svd_desc_score": round(cosine_score * 100, 1) if cosine_score is not None else None
    }
    
    for cocktail, score, jaccard_score, svd_text_score, cosine_score in combined_cocktail_scores[:6]
]

    # if not svd_results:
    #     jaccard_scores = [
    #         (i, jaccard_similarity(script, " ".join(
    #             [c['strIngredient1'], c['strIngredient2'] ] # Add more ingredients as needed
    #         )))
    #         for i, c in enumerate(cocktails)
    #     ]
    #     top_jaccard = sorted(jaccard_scores, key=lambda x: -x[1])[:5]
    #     return jsonify([cocktails[i] for i, _ in top_jaccard])
    # cleaned_results = [clean_cocktail_data(c) for c in svd_results]
    # print(cleaned_results)

    # Recipe SVD
    rec_script_tfidf = recipe_vectorizer.transform([script])
    rec_script_projected = rec_script_tfidf.dot(rec_vt.T)
    rec_similarities = rec_script_projected.dot(recipe_vectors.T)

    # Additional SVD with additional description users put in
    # recipe_desc_similarities = helper_functions.description_svd(recipe_vectorizer, additional_description, rec_vt, recipe_vectors)
    # recipe_desc_similarities = helper_functions.description_svd(recipe_vectorizer_instructions, food_description, i_rec_vt, i_recipe_vectors)

    recipe_jaccard_scores = []
    food_cosine_scores = []
    for recipe in recipes:
        try:
            ingredients_list = ast.literal_eval(recipe['ingredients'])
            ingredients = set(ing.lower().strip() for ing in ingredients_list)
        except (SyntaxError, ValueError):
            ingredients = set()

        jaccard_score = weighted_jaccard_similarity(script_words, ingredients, weight_dict)
        recipe_jaccard_scores.append(jaccard_score)
        
        if food_description is not None:
            query_vec = helper_functions.embed_ingredient_list([food_description], model)
            food_vec = helper_functions.embed_ingredient_list(ingredients_list, model)

            if query_vec is not None and food_vec is not None:
                sim = np.dot(query_vec, food_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(food_vec))
                food_cosine_scores.append(sim)
                print("Similarity:", sim)
            else:
                print("Could not compute similarity")

    # cosine similarity for 
    # cosine_scores = helperfunctions.cosine_similarity(script, recipes, recipe_vectorizer)

    combined_scores = []
    for i, recipe in enumerate(recipes):
        # Get SVD-based scores
        svd_script_score = rec_similarities[0][i]
        # svd_desc_score = None
        cosine_score = None
        if food_description is not None:
            # svd_desc_score = recipe_desc_similarities[0][i]   
            cosine_score = food_cosine_scores[i]         
            combined_svd_score = (1 - beta) * svd_script_score + beta * cosine_score
        else:
            combined_svd_score = svd_script_score

        jaccard_score = recipe_jaccard_scores[i]
        final_score = helper_functions.combine_scores(jaccard_score, combined_svd_score, alpha=0.5)  # Adjust alpha as needed
        combined_scores.append((recipe, final_score, jaccard_score, svd_script_score, cosine_score))

    combined_scores = sorted(combined_scores, key=lambda x: -x[1])
    top_recipes = [
    {
        "data": clean_recipe_data(recipe),
        "score": round(score * 100, 1),
        "jaccard_score": round(jaccard_score * 100, 1),
        "svd_text_score": round(svd_script_score * 100, 1),
        "svd_desc_score": round(cosine_score * 100, 1) if cosine_score is not None else None
    }
    for recipe, score, jaccard_score, svd_script_score, cosine_score in combined_scores[:6]
]
    
    # result = movie_preprocessing.get_movie_foods(movie_title, SCRIPT_FOLDER, FOOD_DATABASE)
    return jsonify({
        "cocktails": top_cocktails,
        # "foods": result["foods"],
        "recipes": top_recipes
    })

@app.route("/movie-suggestions")
def movie_suggestions():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify([])

    try:
        movie_files = [f[:-4] for f in os.listdir(SCRIPT_FOLDER) if f.endswith('.txt')]
    except FileNotFoundError:
        return jsonify([])

    suggestions = [movie for movie in movie_files if query in movie.lower()]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)