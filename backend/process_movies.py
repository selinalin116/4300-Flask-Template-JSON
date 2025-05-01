import os
import csv
import re
import requests
from bs4 import BeautifulSoup
import time

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

def get_decade():
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
                existing_movies.add(base_name)

def delete_untitled():
    # Delete plot files with "untitled" or "film" in their title
    for root, _, files in os.walk(plots_dir):
        for file in files:
            if file.endswith(".txt") and ("untitled" in file.lower() or "film" in file.lower()):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted plot file: {file_path}")

def delete_film_suffix():
    for root, _, files in os.walk(scripts_dir):
        for file in files:
            if file.has("film") in file.lower():
                old_file_path = os.path.join(root, file)
                new_file_name = file.lower().replace("film", "").strip()
                new_file_path = os.path.join(root, new_file_name)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {old_file_path} to {new_file_path}")

def replace_dashes():
    # Iterate through data/scripts and replace dashes and underscores with spaces in filenames
    for root, _, files in os.walk(scripts_dir):
        for file in files:
            if file.endswith(".txt"):
                new_file_name = file.replace("-", " ").replace("_", " ")
                new_file_path = os.path.join(root, new_file_name)
                old_file_path = os.path.join(root, file)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {old_file_path} to {new_file_path}")

BASE_URL = "https://en.wikipedia.org"
LIST_URL = BASE_URL + "/wiki/List_of_Studio_Ghibli_works"
OUTPUT_DIR = "ghibli_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_ghibli_movie_links():
    res = requests.get(LIST_URL)
    soup = BeautifulSoup(res.content, "html.parser")
    links = []

    for td in soup.find_all("td"):
        i = td.find("i")  # Look for <i> tags
        if i:
            a = i.find("a", href=True)  # Find <a> tags inside <i>
            if a and a['href'].startswith("/wiki/") and not a['href'].startswith("/wiki/List"):
                if 'title' in a.attrs:  # Check if 'title' attribute exists
                    links.append((a['title'], BASE_URL + a['href']))
    
    return links

def extract_plot_from_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")

    # Locate the div with id="mw-content-text"
    content_div = soup.find("div", id="mw-content-text")
    if not content_div:
        return None

    # Find all <p> tags within this div
    paragraphs = content_div.find_all("p")
    plot_text = "\n\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

    return plot_text if plot_text else None

def save_plot(title, plot_text):
    # Sanitize and format the filename
    sanitized_name = sanitize_filename(title).replace("-", " ").replace("_", " ")
    filename = f"plot-{sanitized_name}.txt"
    path = os.path.join(scripts_dir, filename)  # Save directly to scripts_dir

    with open(path, "w", encoding="utf-8") as f:
        f.write(plot_text)
    print(f"Saved: {filename}")


# def main():
#     links = get_ghibli_movie_links()
#     print(f"Found {len(links)} possible movie links...")

#     for title, url in links:
#         try:
#             print(f"Fetching plot for: {title}")
#             plot = extract_plot_from_page(url)
#             if plot:
#                 save_plot(title, plot)
#             else:
#                 print(f"⚠️ No plot found for {title}")
#             time.sleep(1)  # Be kind to Wikipedia
#         except Exception as e:
#             print(f"❌ Error processing {title}: {e}")

# if __name__ == "__main__":
#     main()
