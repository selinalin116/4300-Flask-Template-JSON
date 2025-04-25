import os
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import helper_functions

os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))
current_directory = os.path.dirname(os.path.abspath(__file__))

SCRIPT_FOLDER = os.path.join(current_directory, 'data/scripts') 

script_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5)
all_scripts = helper_functions.load_all_scripts(SCRIPT_FOLDER)  # implement this to return list of full scripts
script_tfidf_matrix = script_vectorizer.fit_transform(all_scripts)

k = 40  # latent dimensions
U_s, s_s, vt_s = svds(script_tfidf_matrix, k=k)
script_vectors = normalize(U_s)  # latent movie embeddings
