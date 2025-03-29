from cocktail import *
from recipe import recipe_vectors, recipes, clean_recipe_data, rec_vt, recipe_vectorizer
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import helper_functions

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

current_directory = os.path.dirname(os.path.abspath(__file__))

SCRIPT_FOLDER = os.path.join(current_directory, 'data/scripts')  
# FOOD_DATABASE = os.path.join(current_directory, 'data/recipes_names.csv')  
FOOD_DATABASE = os.path.join(current_directory, 'database.txt')  


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
    movie_title = request.args.get('movie', '').strip()
    
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"})
    
    additional_description = None
    if 'description' in request.args:
        additional_description = request.args.get('description').strip()
    
    script = helper_functions.get_movie_script(movie_title, SCRIPT_FOLDER)
    if not script:
        return jsonify({"error": "Script not found"})
    
    script_tfidf = cocktail_vectorizer.transform([script])
    script_projected = script_tfidf.dot(vt.T)
    script_projected = normalize(script_projected)
    
    script_words = set(script.split())
    
    # Cocktail SVD
    similarities = script_projected.dot(cocktail_vectors.T)
    
    # Jaccard similairty for cocktails
    cocktail_jaccard_scores = []
    for cocktail in cocktails:
        cocktail_ingredients = set(cocktail.get('ingredients', []))  # Ensure ingredients are a list
        jaccard_score = helper_functions.jaccard_similarity(script_words, cocktail_ingredients)
        cocktail_jaccard_scores.append(jaccard_score)
    
    # Combine jaccard similarity for cocktails
    combined_cocktail_scores = []
    for i, cocktail in enumerate(cocktails):
        svd_score = similarities[0][i]
        jaccard_score = cocktail_jaccard_scores[i]
        combined_score = helper_functions.combine_scores(jaccard_score, svd_score, alpha=0.5)  # Adjust alpha as needed
        combined_cocktail_scores.append((cocktail, combined_score))
    
    # Sort and Get Top Cocktails
    combined_cocktail_scores = sorted(combined_cocktail_scores, key=lambda x: -x[1])
    top_cocktails = [clean_cocktail_data(cocktail) for cocktail, score in combined_cocktail_scores[:6]]


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
    desc_similarities = None
    if additional_description:
        additional_tfidf = recipe_vectorizer.transform([additional_description])
        additional_projected = additional_tfidf.dot(rec_vt.T)
        desc_similarities = additional_projected.dot(recipe_vectors.T)

    recipe_jaccard_scores = []
    for recipe in recipes:
        recipe_ingredients = set(recipe['ingredients'])
        jaccard_score = helper_functions.jaccard_similarity(script_words, recipe_ingredients)
        recipe_jaccard_scores.append(jaccard_score)

    # cosine similarity for 
    # cosine_scores = helperfunctions.cosine_similarity(script, recipes, recipe_vectorizer)

    beta = 0.3

    combined_scores = []
    for i, recipe in enumerate(recipes):
        # Get SVD-based scores
        svd_script_score = rec_similarities[0][i]
        
        if desc_similarities is not None:
            svd_desc_score = desc_similarities[0][i]

            # Weighted sum of script and description SVD scores
            combined_svd_score = (1 - beta) * svd_script_score + beta * svd_desc_score
        else:
            combined_svd_score = svd_script_score

        jaccard_score = recipe_jaccard_scores[i]
        # cosine_score = cosine_scores[i]
        final_score = helper_functions.combine_scores(jaccard_score, combined_svd_score, alpha=0.5)  # Adjust alpha as needed
        combined_scores.append((recipe, final_score))

    combined_scores = sorted(combined_scores, key=lambda x: -x[1])
    top_recipes = [clean_recipe_data(recipe) for recipe, score in combined_scores[:6]]

    # result = movie_preprocessing.get_movie_foods(movie_title, SCRIPT_FOLDER, FOOD_DATABASE)
    return jsonify({
        "cocktails": top_cocktails,
        # "foods": result["foods"],
        "recipes": top_recipes
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)