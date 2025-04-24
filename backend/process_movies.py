import os
import csv
import re

scripts_dir = "4300-Flask-Template-JSON/backend/data/scripts"
plots_dir = os.path.join("4300-Flask-Template-JSON/backend/data/plots")

os.makedirs(scripts_dir, exist_ok=True)

movies_by_decade = {
    "2000s": "4300-Flask-Template-JSON/backend/2000s-movies.csv",
    "2010s": "4300-Flask-Template-JSON/backend/2010s-movies.csv",
    "2020s": "4300-Flask-Template-JSON/backend/2020s-movies.csv"
}

existing_movies = {
    re.sub(r'\s+\d{4}$', '', file[5:-4].replace("-", " ").lower())  # Remove "plot-", ".txt", dashes, and years
    for file in os.listdir(scripts_dir) if file.startswith("plot-") and file.endswith(".txt")
}

def sanitize_filename(name):
    return re.sub(r'[^\w\s-]', '', name).replace(' ', '_')

for decade, movies_file in movies_by_decade.items():
    with open(movies_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            movie_name = row["title"].strip()
            movie_plot = row["plot"].strip()

            # Correct movie titles like "Great Gatsby, The" to "The Great Gatsby"
            if "," in movie_name:
                parts = movie_name.split(", ")
                if len(parts) == 2:
                    movie_name = f"{parts[1]} {parts[0]}"

            base_name = re.sub(r'\s+\d{4}$', '', movie_name).lower()

            # skip if the movie (ignoring year) is already in data/scripts
            if base_name in existing_movies:
                print(f"Skipping {movie_name}, already in data/scripts.")
                continue

            # Sanitize the movie name for filenames
            sanitized_name = sanitize_filename(movie_name)

            script_file_path = os.path.join(scripts_dir, f"plot-{sanitized_name}.txt")
            with open(script_file_path, "w", encoding="utf-8") as script_file:
                script_file.write(movie_plot)
            print(f"Saved plot for {movie_name} to {script_file_path}.")
            existing_movies.add(base_name)  # Add to existing_movies to prevent duplicates

# Delete plot files with "untitled" or "film" in their title
for root, _, files in os.walk(plots_dir):
    for file in files:
        if file.endswith(".txt") and ("untitled" in file.lower() or "film" in file.lower()):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Deleted plot file: {file_path}")

