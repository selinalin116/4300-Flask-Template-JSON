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
from recipe import recipe_vectors, recipes, clean_recipe_data
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import movie_preprocessing
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
    
    script_tfidf = vectorizer.transform([script])
    script_projected = script_tfidf.dot(vt.T)
    script_projected = normalize(script_projected)
    
    similarities = script_projected.dot(cocktail_vectors.T)
    top_indices = np.argsort(-similarities[0])[:3]
    svd_results = [cocktails[i] for i in top_indices]
    # if not svd_results:
    #     jaccard_scores = [
    #         (i, jaccard_similarity(script, " ".join(
    #             [c['strIngredient1'], c['strIngredient2'] ] # Add more ingredients as needed
    #         )))
    #         for i, c in enumerate(cocktails)
    #     ]
    #     top_jaccard = sorted(jaccard_scores, key=lambda x: -x[1])[:5]
    #     return jsonify([cocktails[i] for i, _ in top_jaccard])
    cleaned_results = [clean_cocktail_data(c) for c in svd_results]
    # print(cleaned_results)

    # Recipe SVD
    rec_similarities = script_projected.dot(recipe_vectors.T)
    rec_top_indices = np.argsort(-rec_similarities[0])[:3]
    rec_svd_results = [recipes[i] for i in rec_top_indices]
    recipe_results = [clean_recipe_data(recipe) for recipe in rec_svd_results]

    result = movie_preprocessing.get_movie_foods(movie_title, SCRIPT_FOLDER, FOOD_DATABASE)
    return jsonify({
        "cocktails": cleaned_results,
        "foods": result["foods"],
        "recipes": recipe_results
    })

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)