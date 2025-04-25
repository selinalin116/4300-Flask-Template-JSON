import os
import re

SCRIPT_FOLDER = "/home/irisho/4300-Flask-Template-JSON/backend/data/scripts"

def get_base_name(filename):
    base_name = filename.lower().replace(".txt", "")  # Remove .txt
    if base_name.startswith("plot-"):
        base_name = base_name[5:]  # Remove "plot-"
    base_name = re.sub(r'[-_]', ' ', base_name)  # Normalize separators
    base_name = re.sub(r'\b(film|movie)\b', '', base_name)
    base_name = re.sub(r'\b\d{4}\b', '', base_name)  # Remove years
    base_name = re.sub(r'\s+', ' ', base_name).strip()
    return base_name

def remove_duplicate_scripts(directory):
    seen_movies = {}
    duplicates = []

    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(".txt"):
            base_name = get_base_name(filename)
            if base_name in seen_movies:
                # Only keep the shorter (cleaner) filename
                existing = seen_movies[base_name]
                if len(filename) < len(existing):
                    # Replace with shorter version
                    duplicates.append(existing)
                    seen_movies[base_name] = filename
                else:
                    duplicates.append(filename)
            else:
                seen_movies[base_name] = filename

    for duplicate in duplicates:
        duplicate_path = os.path.join(directory, duplicate)
        os.remove(duplicate_path)
        print(f"Removed duplicate: {duplicate_path}")


if __name__ == "__main__":
    remove_duplicate_scripts(SCRIPT_FOLDER)
