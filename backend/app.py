# import json
# import os
# from flask import Flask, render_template, request
# from flask_cors import CORS
# from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
# import pandas as pd
# import movie_preprocessing

# # ROOT_PATH for linking with all your files. 
# # Feel free to use a config.py or settings.py with a global export variable
# os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# # Get the directory of the current script
# current_directory = os.path.dirname(os.path.abspath(__file__))

# # Specify the path to the JSON file relative to the current script
# json_file_path = os.path.join(current_directory, 'init.json')

# # Assuming your JSON data is stored in a file named 'init.json'
# with open(json_file_path, 'r') as file:
#     data = json.load(file)
#     episodes_df = pd.DataFrame(data['episodes'])
#     reviews_df = pd.DataFrame(data['reviews'])

# app = Flask(__name__)
# CORS(app)

# # Sample search using json with pandas
# def json_search(query):
#     matches = []
#     merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
#     matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
#     matches_filtered = matches[['title', 'descr', 'imdb_rating']]
#     matches_filtered_json = matches_filtered.to_json(orient='records')
#     return matches_filtered_json

# @app.route("/")
# def home():
#     return render_template('base.html',title="sample html")

# @app.route("/episodes")
# def episodes_search():
#     text = request.args.get("title")
#     return json_search(text)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)
# app.py

# from cocktail import *
# import requests
# from recipe import *
# import os
# from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS
# import movie_preprocessing
# import helperfunctions

# os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# current_directory = os.path.dirname(os.path.abspath(__file__))

# SCRIPT_FOLDER = os.path.join(current_directory, 'data/scripts')  
# FOOD_DATABASE = os.path.join(current_directory, 'database.txt')  

# app = Flask(__name__)
# CORS(app)

# @app.route("/")
# def home():
#     return render_template('base.html', title="Movie Food Finder")

# @app.route("/find-foods")
# def find_foods():
#     """API endpoint to find foods in a movie script"""
#     movie_title = request.args.get('movie', '').strip()
    
#     if not movie_title:
#         return jsonify({"error": "Please enter a movie title"})
    
#     script = helperfunctions.get_movie_script(movie_title, SCRIPT_FOLDER)
#     if not script:
#         return jsonify({"error": "Script not found"})
    
#     script_tfidf = vectorizer.transform([script])
#     script_projected = script_tfidf.dot(vt.T)
#     script_projected = normalize(script_projected)
    
#     # Cocktail SVD
#     similarities = script_projected.dot(cocktail_vectors.T)
#     top_indices = np.argsort(-similarities[0])[:3]
#     svd_results = [cocktails[i] for i in top_indices]
#     # if not svd_results:
#     #     jaccard_scores = [
#     #         (i, jaccard_similarity(script, " ".join(
#     #             [c['strIngredient1'], c['strIngredient2'] ] # Add more ingredients as needed
#     #         )))
#     #         for i, c in enumerate(cocktails)
#     #     ]
#     #     top_jaccard = sorted(jaccard_scores, key=lambda x: -x[1])[:5]
#     #     return jsonify([cocktails[i] for i, _ in top_jaccard])
#     cleaned_results = [clean_cocktail_data(c) for c in svd_results]
#     # print(cleaned_results)

#     # Recipe SVD
#     # similarities = script_projected.dot(recipe_vectors.T)
#     # top_indices = np.argsort(-similarities[0])[:3]
#     # svd_results = [recipes[i] for i in top_indices]

#     result = movie_preprocessing.get_movie_foods(movie_title, SCRIPT_FOLDER, FOOD_DATABASE)
    
#     return jsonify({
#         "cocktails": cleaned_results,
#         "foods": result["foods"]
#     })

# # if __name__ == '__main__':
# #     app.run(debug=True, host="0.0.0.0", port=5000)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)

import requests
from cocktail import *
from recipe import recipe_vectors, recipes, clean_recipe_data, rec_vt, recipe_vectorizer
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import helperfunctions

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
    """API endpoint to find foods in a movie script"""
    movie_title = request.args.get('movie', '').strip()
    
    if not movie_title:
        return jsonify({"error": "Please enter a movie title"})
    
    script = helperfunctions.get_movie_script(movie_title, SCRIPT_FOLDER)
    if not script:
        return jsonify({"error": "Script not found"})
    
    # vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
    script_tfidf = cocktail_vectorizer.transform([script])
    script_projected = script_tfidf.dot(vt.T)
    script_projected = normalize(script_projected)
    
    script_words = set(script.split())
    
    # cocktail SVD
    similarities = script_projected.dot(cocktail_vectors.T)
    
    # jaccard similairty for cocktails
    cocktail_jaccard_scores = []
    for cocktail in cocktails:
        cocktail_ingredients = set(cocktail.get('ingredients', []))  # Ensure ingredients are a list
        jaccard_score = helperfunctions.jaccard_similarity(script_words, cocktail_ingredients)
        cocktail_jaccard_scores.append(jaccard_score)
    
    # combine jaccard similarity for cocktails
    combined_cocktail_scores = []
    for i, cocktail in enumerate(cocktails):
        svd_score = similarities[0][i]
        jaccard_score = cocktail_jaccard_scores[i]
        combined_score = helperfunctions.combine_scores(jaccard_score, svd_score, alpha=0.5)  # Adjust alpha as needed
        combined_cocktail_scores.append((cocktail, combined_score))
    
    # Sort and Get Top Cocktails
    combined_cocktail_scores = sorted(combined_cocktail_scores, key=lambda x: -x[1])
    top_cocktails = [clean_cocktail_data(cocktail) for cocktail, score in combined_cocktail_scores[:3]]


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
    # TODO: script_projected probably needs to be redone here
    rec_script_tfidf = recipe_vectorizer.transform([script])
    rec_script_projected = rec_script_tfidf.dot(rec_vt.T)
    rec_similarities = rec_script_projected.dot(recipe_vectors.T)
    rec_top_indices = np.argsort(-rec_similarities[0])[:3]
    rec_svd_results = [recipes[i] for i in rec_top_indices]
    recipe_results = [clean_recipe_data(recipe) for recipe in rec_svd_results]

    recipe_jaccard_scores = []
    for recipe in recipes:
        recipe_ingredients = set(recipe['ingredients'])
        jaccard_score = helperfunctions.jaccard_similarity(script_words, recipe_ingredients)
        recipe_jaccard_scores.append(jaccard_score)

    combined_scores = []
    for i, recipe in enumerate(recipes):
        svd_score = rec_similarities[0][i]
        jaccard_score = recipe_jaccard_scores[i]
        combined_score = helperfunctions.combine_scores(jaccard_score, svd_score, alpha=0.5)  # Adjust alpha as needed
        combined_scores.append((recipe, combined_score))

    combined_scores = sorted(combined_scores, key=lambda x: -x[1])
    top_recipes = [clean_recipe_data(recipe) for recipe, score in combined_scores[:3]]

    # result = movie_preprocessing.get_movie_foods(movie_title, SCRIPT_FOLDER, FOOD_DATABASE)
    return jsonify({
        "cocktails": top_cocktails,
        # "foods": result["foods"],
        "recipes": top_recipes
    })

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)