import os
import csv

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, 'data/movie_synopsis.csv')
scripts_folder = os.path.join(current_directory, 'data/scripts')

os.makedirs(scripts_folder, exist_ok=True)

with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        movie_name = row['title'].replace(":", "").replace("/", "").replace("\\", "").replace("?", "").replace("*", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "").strip()
        synopsis = row['synopsis']
        file_path = os.path.join(scripts_folder, f"{movie_name}.txt")
        
        with open(file_path, mode='w', encoding='utf-8') as txt_file:
            txt_file.write(synopsis)

print(f"Movie summaries have been written to {scripts_folder}")
