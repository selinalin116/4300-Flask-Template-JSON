from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import json
import zipfile
import ast

zip_filename = 'data/recipes_cleaned.zip'
file_inside_zip = 'recipes_cleaned.json'

# Open the ZIP file and read the file inside it
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    with zipf.open(file_inside_zip) as file:
        data = file.read().decode('utf-8')

recipes = json.loads(data)

def parse_steps_to_paragraph(steps_string):
    """
    Converts a string representation of a list of steps into a paragraph.
    """
    try:
        # Remove the outer quotes if present
        if steps_string.startswith('"') and steps_string.endswith('"'):
            steps_string = steps_string[1:-1]
            
        steps_list = ast.literal_eval(steps_string)
        
        paragraph = " ".join(steps_list)
        
        paragraph = paragraph.replace("&amp;", "&")
        
        return paragraph
    except (SyntaxError, ValueError):
        cleaned = steps_string.strip("[]'\"")
        steps = [step.strip(" '\"") for step in cleaned.split("', '")]
        return " ".join(steps)
    
recipe_instructions = [parse_steps_to_paragraph(i['steps']) for i in recipes]

recipe_descriptions = [i['description'] for i in recipes]

recipe_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
recipe_tfidf = recipe_vectorizer.fit_transform(recipe_descriptions)

k = 40  # recipesvd has a graph that explains this choice
u, s, rec_vt = svds(recipe_tfidf, k=k)
recipe_vectors = normalize(u, axis=1)

combined_texts = [
    f"{desc} {instr}"
    for desc, instr in zip(recipe_descriptions, recipe_instructions)
]

recipe_vectorizer_instructions = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
recipe_tfidf_instructions = recipe_vectorizer_instructions.fit_transform(combined_texts)

i_u, i_s, i_rec_vt = svds(recipe_tfidf_instructions, k=k)
i_recipe_vectors = normalize(i_u, axis=1)

def clean_recipe_data(recipe):
    """
    Extracts and formats key details from a recipe dictionary.
    """
    instructions = parse_steps_to_paragraph(recipe["steps"])

    return {
        'name': recipe['name'],
        'description': recipe['description'],
        'ingredients': ast.literal_eval(recipe['ingredients_raw_str']),
        'rating': recipe['average_rating'],
        'instructions': instructions
        
    }