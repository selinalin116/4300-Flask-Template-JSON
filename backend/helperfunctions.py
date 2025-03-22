import os

def get_movie_script(movie_title, folder):
    """
    Load and return the content of a movie script.
    Supports exact and partial matching with hyphens or underscores.
    """
    base = movie_title.lower()
    variations = [
        f"{base.replace(' ', '-')}.txt",
        f"{base.replace(' ', '_')}.txt"
    ]
    
    # Try exact filename matches first
    for filename in variations:
        script_path = os.path.join(folder, filename)
        if os.path.exists(script_path):
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()

    # Fallback: try partial match
    for file in os.listdir(folder):
        if file.endswith('.txt'):
            filename_lower = file.lower()
            # Check if any word from input appears in filename
            input_words = base.split()
            if all(word in filename_lower for word in input_words):
                script_path = os.path.join(folder, file)
                with open(script_path, 'r', encoding='utf-8') as f:
                    return f.read()
    
    return None
