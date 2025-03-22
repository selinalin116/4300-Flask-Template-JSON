import os

def find_script_file(movie_title, script_folder):
    """
    Find the script file for a given movie title.
    Handles various formats including spaces vs. hyphens.
    """
    movie_title_lower = movie_title.lower()
    
    # Create variations of the movie title for matching
    variations = [
        movie_title_lower,                      # Original (e.g., "star wars")
        movie_title_lower.replace(' ', '-'),    # Spaces to hyphens (e.g., "star-wars")
        movie_title_lower.replace(' ', '_'),    # Spaces to underscores (e.g., "star_wars")
        movie_title_lower.replace('-', ' '),    # Hyphens to spaces
        movie_title_lower.replace('_', ' '),    # Underscores to spaces
        ''.join(movie_title_lower.split())      # No spaces (e.g., "starwars")
    ]
    
    # Try exact match first
    for filename in os.listdir(script_folder):
        if filename.endswith('.txt'):
            filename_without_ext = os.path.splitext(filename)[0].lower()
            
            # Check if any variation matches the filename
            for variation in variations:
                if filename_without_ext == variation:
                    return os.path.join(script_folder, filename)
    
    # Try partial match if exact match fails
    for filename in os.listdir(script_folder):
        if filename.endswith('.txt'):
            filename_without_ext = os.path.splitext(filename)[0].lower()
            
            # Check if any variation is contained in the filename
            for variation in variations:
                if variation in filename_without_ext:
                    return os.path.join(script_folder, filename)
    
    # If no match is found
    return None