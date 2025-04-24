from cocktail import *
from recipe import recipe_vectors, recipes, clean_recipe_data, rec_vt, recipe_vectorizer, i_rec_vt, i_recipe_vectors, recipe_vectorizer_instructions, recipe_compute_idf
import os, string
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import helper_functions
import ast
from cocktail import extract_ingredients
from helper_functions import dietary_res, drinks_filtered, penalize_jaccard_similarity, cosine_sim, normalize_ingredient, get_script_phrases
from gensim.models import KeyedVectors
import numpy as np
from pairings import get_pairing_score_ranked
from collections import defaultdict
import ast
# from scipy.linalg import orthogonal_procrustes

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

SCRIPT_FOLDER = os.path.join(current_directory, 'data/scripts')  
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
    model = KeyedVectors.load("model/glove-wiki.kv")  

    movie_title = request.args.get('movie', '').strip()
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"})

    drink_description = request.args.get('drinkdescription')
    if drink_description is not None:
        drink_description = drink_description.strip().lower()
    
    food_description = request.args.get('fooddescription')
    if food_description is not None:
        food_description = food_description.strip().lower()

    dietary_restrictions = request.args.get('dietary', '').strip()
    dietary_restrictions = [r.strip() for r in dietary_restrictions.split(',')] if dietary_restrictions else []

    alcohol_preference = request.args.get('alcohol', '').strip()
    alcohol_preference = [a.strip().lower().replace('-', ' ') for a in alcohol_preference.split(',')] if alcohol_preference else []

    script = helper_functions.get_movie_script(movie_title, SCRIPT_FOLDER)
    if not script:
        return jsonify({"error": "Script or plot not found"})
    
    script_tfidf = cocktail_vectorizer.transform([script])
    script_projected = script_tfidf.dot(vt.T)
    script_projected = normalize(script_projected)
    
    script_words = set(script.split())
    # script_words = get_script_phrases(script)

    # Extract cocktail ingredients
    cocktail_ingredients_set = set()
    for cocktail in cocktails:
        cocktail_ingredients = extract_ingredients(cocktail)
        for ingredient in cocktail_ingredients:
            normalized_results = normalize_ingredient(ingredient)
            if normalized_results:
                for result in normalized_results:
                    cocktail_ingredients_set.add(result)

    # Extract recipe ingredients
    recipe_ingredients_set = set()
    for recipe in recipes:
        try:
            ingredients_list = ast.literal_eval(recipe['ingredients'])
            # recipe_ingredients_set.update(ing.lower().strip() for ing in ingredients_list)
            for ing in ingredients_list:
                phrase = ing.lower().strip()
                words = phrase.split()

            # Add full ingredient
            recipe_ingredients_set.add(phrase)

            # Add last two words individually
            if len(words) >= 2:
                recipe_ingredients_set.update(words[-2:])
            elif words:
                recipe_ingredients_set.add(words[0])
        except (SyntaxError, ValueError):
            pass

    similarities = script_projected.dot(cocktail_vectors.T)

    beta = 0.9

    cocktail_desc_similarities = helper_functions.description_svd(cocktail_vectorizer, drink_description, vt, cocktail_vectors)
    source = "script"
    weight_dict = {word: 1.5 for word in script_words}
    cocktail_tfidf = compute_idf()
    food_tfidf = recipe_compute_idf()

    cocktail_weight_dict = helper_functions.build_combined_weight_dict(script_words, cocktail_tfidf, boost=1.5)

    food_weight_dict = helper_functions.build_combined_weight_dict(script_words, food_tfidf, boost=1.5)

    cocktail_jaccard_scores = []
    cocktail_raw_jaccard_scores = []
    cocktail_cosine_scores = []
    for cocktail in cocktails:
        cocktail_ingredients = extract_ingredients(cocktail)
        cocktail_vec = helper_functions.embed_ingredient_list(cocktail_ingredients, model)

        cocktail_ingredients_set = set(" ".join(cocktail_ingredients).lower().split())
        # jaccard_score = weighted_jaccard_similarity(script_words, cocktail_ingredients_set, cocktail_weight_dict)
        jaccard_score, raw_jaccard_score = penalize_jaccard_similarity(script_words, cocktail_ingredients_set, cocktail_weight_dict, weight_dict)
        cocktail_jaccard_scores.append(jaccard_score)
        cocktail_raw_jaccard_scores.append(raw_jaccard_score)
        # cosine_score = helper_functions.cosine_similarity(cocktail_ingredients_tfidf, drink_description, cocktail_ingredients_vectorizer)
        # print(cosine_score)

        if drink_description is not None:
            query_vec = helper_functions.embed_ingredient_list([drink_description], model)

            if query_vec is not None and cocktail_vec is not None:
                sim = cosine_sim(query_vec, cocktail_vec)
                cocktail_cosine_scores.append(sim)
                # print("Similarity:", sim)
            # else:
            #     print("Could not compute similarity")

    
    combined_cocktail_scores = []
    for i, cocktail in enumerate(cocktails):
        cocktail_ingredients = set(" ".join(extract_ingredients(cocktail)).lower().split())
        intersecting_words = script_words & cocktail_ingredients  # Compute intersecting words

        # Compute top SVD terms
        top_svd_indices = np.argsort(-script_projected[0])[:5]
        top_svd_terms = [cocktail_vectorizer.get_feature_names_out()[i] for i in top_svd_indices]

        svd_text_score = similarities[0][i]
        combined_desc_score = None
        if drink_description is not None:
            svd_desc_score = cocktail_desc_similarities[0][i]
            cosine_score = cocktail_cosine_scores[i]
            alpha = 0.8
            combined_desc_score = (1 - alpha) * svd_desc_score + alpha * cosine_score
            combined_svd_score = (1 - beta) * svd_text_score + beta * combined_desc_score
        else:
            combined_svd_score = svd_text_score

        jaccard_score = cocktail_jaccard_scores[i]
        raw_jaccard_score = cocktail_raw_jaccard_scores[i]

        # boost score if user preference matches cocktail ingredients
        preference_boost = 0
        if drink_description:
            preference_boost = sum(1 for word in drink_description.split() if word in cocktail_ingredients) * 0.1

        combined_score = helper_functions.combine_scores(jaccard_score, combined_svd_score, alpha=0.4) + preference_boost
        combined_cocktail_scores.append((cocktail, combined_score, raw_jaccard_score, svd_text_score, combined_desc_score, intersecting_words, top_svd_terms, source))

    combined_cocktail_scores = sorted(combined_cocktail_scores, key=lambda x: -x[1])

    # if (len(dietary_restrictions)>0):
    #     combined_cocktail_scores = dietary_res(combined_cocktail_scores, 6, dietary_restrictions)

    # Filter based on user preferences
    if (len(alcohol_preference)==1):
        combined_cocktail_scores = drinks_filtered(combined_cocktail_scores, 6, alcohol_preference)
    
    # Sort and Get Top Cocktails
    top_cocktails = [
        {
            "data": clean_cocktail_data(cocktail),
            "score": round(score * 100, 1),
            "jaccard_score": round(raw_jaccard_score * 100, 1),
            "svd_text_score": round(svd_text_score * 100, 1),
            "svd_desc_score": round(combined_desc_score * 100, 1) if combined_desc_score is not None else None,
            "jaccard_intersection": list(intersecting_words),
            "top_svd_terms": top_svd_terms,
            "source": source
        }
        for cocktail, score, raw_jaccard_score, svd_text_score, combined_desc_score, intersecting_words, top_svd_terms, source in combined_cocktail_scores[:6]
    ]

    rec_script_tfidf = recipe_vectorizer.transform([script])
    rec_script_projected = rec_script_tfidf.dot(rec_vt.T)
    rec_similarities = rec_script_projected.dot(recipe_vectors.T)

    recipe_desc_similarities = helper_functions.description_svd(recipe_vectorizer_instructions, food_description, i_rec_vt, i_recipe_vectors)

    recipe_jaccard_scores = []
    food_raw_jaccard_scores = []
    food_cosine_scores = []
    for recipe in recipes:
        try:
            ingredients = set()
            ingredients_list = ast.literal_eval(recipe['ingredients'])
            for ing in ingredients_list:
                ing = normalize_ingredient(ing)
                print(ing)
                words = ing

                ingredients.add(phrase)

                if len(words) >= 2:
                    ingredients.update(words[-2:])
                elif words:
                    ingredients.add(words[0])
        except (SyntaxError, ValueError):
            ingredients = set()

        # jaccard_score = weighted_jaccard_similarity(script_words, ingredients, weight_dict)
        jaccard_score, raw_jaccard_score = penalize_jaccard_similarity(script_words, ingredients, food_weight_dict, weight_dict)
        recipe_jaccard_scores.append(jaccard_score)
        food_raw_jaccard_scores.append(raw_jaccard_score)
        
        if food_description is not None:
            query_vec = helper_functions.embed_ingredient_list([food_description], model)
            # print(query_vec)
            food_vec = helper_functions.embed_ingredient_list(ingredients_list, model)
            # print(food_vec)

            if query_vec is not None and food_vec is not None:
                sim = cosine_sim(query_vec, food_vec)
                food_cosine_scores.append(sim)
            # else:
            #     print("Could not compute similarity")

    combined_scores = []
    for i, recipe in enumerate(recipes):
        try:
            ingredients = set()
            ingredients_list = ast.literal_eval(recipe['ingredients'])
            for ing in ingredients_list:
                ing = normalize_ingredient(ing)
                words = ing

                ingredients.add(phrase)

                if len(words) >= 2:
                    ingredients.update(words[-2:])
                elif words:
                    ingredients.add(words[0])

            # print(recipe_ingredients_set)
        except (SyntaxError, ValueError):
            ingredients = set()

        intersecting_words = script_words & ingredients  # Compute intersecting words

        # Compute top SVD terms
        top_svd_indices = np.argsort(-rec_script_projected[0])[:5]
        top_svd_terms = [recipe_vectorizer.get_feature_names_out()[i] for i in top_svd_indices]


        svd_script_score = rec_similarities[0][i]
        combined_desc_score = None
        if food_description is not None:
            svd_desc_score = recipe_desc_similarities[0][i]   
            cosine_score = food_cosine_scores[i]   
            alpha = 0.8 
            combined_desc_score = (1 - alpha) * svd_desc_score + alpha * cosine_score     
            combined_svd_score = (1 - beta) * svd_script_score + beta * combined_desc_score
        else:
            combined_svd_score = svd_script_score

        jaccard_score = recipe_jaccard_scores[i]
        raw_jaccard_score = food_raw_jaccard_scores[i]

        # boost score if user preference matches recipe ingredients
        preference_boost = 0
        if food_description:
            preference_boost = sum(1 for word in food_description.split() if word in ingredients) * 0.1

        base_score = helper_functions.combine_scores(jaccard_score, combined_svd_score, alpha=0.4) + preference_boost

        rating = recipe.get("average_rating", 0) or 0
        normalized_rating = rating / 5.0  
        final_score = (0.95 * base_score) + (0.05 * normalized_rating)

        combined_scores.append((recipe, final_score, raw_jaccard_score, jaccard_score, svd_script_score, combined_desc_score, base_score, intersecting_words, top_svd_terms))

        # combined_scores.append((recipe, final_score, jaccard_score, svd_script_score, combined_desc_score, base_score, intersecting_words, top_svd_terms))

    combined_scores = sorted(
        combined_scores,
        key=lambda x: (
            -x[1],  # primary sorting: score
            -(x[0].get("average_rating", 0) or 0)  # secondary sorting: rating from recipe
        )
    )
    if (len(dietary_restrictions)>0):
        combined_scores = dietary_res(combined_scores, 6, dietary_restrictions)
    top_recipes = [
        {
            "data": clean_recipe_data(recipe),
            "score": round(score * 100, 1),
            # "jaccard_score": round(jaccard_score * 100, 1),
            "jaccard_score": round(raw_jaccard_score * 100, 1),
            "weighted_jaccard": round(jaccard_score * 100, 1),
            "svd_text_score": round(svd_script_score * 100, 1),
            "svd_desc_score": round(combined_desc_score * 100, 1) if combined_desc_score is not None else None,
            "jaccard_intersection": list(intersecting_words),
            "top_svd_terms": top_svd_terms
            }
        for recipe, _, raw_jaccard_score, jaccard_score, svd_script_score, combined_desc_score, score, intersecting_words, top_svd_terms in combined_scores[:6]
    ]
    

    pairings_by_recipe = defaultdict(list)

    for cocktail in top_cocktails:
        cocktail_ings = cocktail["data"]["ingredients"]

        for recipe in top_recipes:
            recipe_name = recipe["data"]["name"]
            recipe_ings = recipe["data"]["ingredients"]
            if isinstance(recipe_ings, str):
                recipe_ings = ast.literal_eval(recipe_ings)

            label, rank = get_pairing_score_ranked(cocktail_ings, recipe_ings, model)

            if rank > 0:
                pairings_by_recipe[recipe_name].append({
                    "cocktail": cocktail["data"]["name"],
                    "link": cocktail["data"]["recipe_link"],
                    "compatibility": label,
                    "rank": rank
                })

    for recipe in top_recipes:
        recipe_name = recipe["data"]["name"]
        pairings = pairings_by_recipe.get(recipe_name, [])

        sorted_pairings = sorted(pairings, key=lambda x: -x["rank"])[:3]

        recipe["recommended_cocktails"] = sorted_pairings
    
    return jsonify({
        "cocktails": top_cocktails,
        "recipes": top_recipes,
        "source":source
    })

@app.route("/movie-suggestions")
def movie_suggestions():
    query = request.args.get('query', '').strip().lower()
    if len(query) < 3: 
        return jsonify([])
    try:
        movie_files = [
            f[5:-4].replace('-', ' ').replace('_', ' ')
            if f.startswith("plot-") else f[:-4].replace('-', ' ').replace('_', ' ')
            for f in os.listdir(SCRIPT_FOLDER) if f.endswith('.txt')
        ]
    except FileNotFoundError:
        return jsonify([])

    suggestions = [movie for movie in movie_files if query in movie.lower()]
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

# # if 'DB_NAME' not in os.environ:
# #     app.run(debug=True,host="0.0.0.0",port=5000)