import os
import csv

current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_directory, 'data/movie_synopsis.csv')
scripts_folder = os.path.join(current_directory, 'data/scripts')

os.makedirs(scripts_folder, exist_ok=True)

# with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
#     reader = csv.DictReader(csv_file)
#     for row in reader:
#         movie_name = row['title'].replace(":", "").replace("/", "").replace("\\", "").replace("?", "").replace("*", "").replace("\"", "").replace("<", "").replace(">", "").replace("|", "").strip()
#         synopsis = row['synopsis']
#         file_path = os.path.join(scripts_folder, f"{movie_name}.txt")
        
#         with open(file_path, mode='w', encoding='utf-8') as txt_file:
#             txt_file.write(synopsis)

# print(f"Movie summaries have been written to {scripts_folder}")

def delete_short_movie_files(folder_path, min_word_count=200):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                word_count = len(content.split())
                if word_count < min_word_count:
                    os.remove(file_path)
                    print(f"Deleted {file_path} (word count: {word_count})")

delete_short_movie_files(scripts_folder, min_word_count=200)
