import os
import re
import pandas as pd

# Define directory and file paths
DIR_PATH = os.path.join("..", "data")
INPUT_FILE = "dataset_yt_video_details.txt"
OUTPUT_FILE = "dataset_yt_video_details.csv"

INPUT_FILE_PATH = os.path.join(DIR_PATH, INPUT_FILE)
OUTPUT_FILE_PATH = os.path.join(DIR_PATH, OUTPUT_FILE)

def sanitize_filename(filename):
    """
    Sanitize the filename by replacing illegal characters with a hyphen and removing leading/trailing periods.
    """
    sanitized_filename = re.sub(r'[\\/:*?"<>|]', '-', filename)
    sanitized_filename = re.sub(r'^\.+|\.+$', '', sanitized_filename)
    return sanitized_filename

def parse_film_data(text):
    """
    Parse film data from text using regex patterns and return a list of dictionaries with film details.
    """
    film_pattern = re.compile(
        r'film_name="(?P<name>[^"]+)"\n'
        r'film_year=(?P<year>\d+)\n'
        r'film_genre="(?P<genre>[^"]+)"\n'
        r'film_url="(?P<url>[^"]+)"'
    )
    
    films = []
    
    for match in film_pattern.finditer(text):
        film = match.groupdict()
        film_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', film['url'])
        film['video_id'] = film_id_match.group(1) if film_id_match else None
        film['sanitized_name'] = sanitize_filename(film['name'])
        films.append(film)
    
    return films

def save_films_to_csv(films, output_file):
    """
    Save the list of films to a CSV file.
    """
    if not films:
        print("No films to save")
        return
    
    df = pd.DataFrame(films)
    df.to_csv(output_file, index=False)
    print(f"Saved film details to {output_file}")

if __name__ == "__main__":
    try:
        with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as file:
            text = file.read()
        
        films = parse_film_data(text)
        print(f"Total films parsed: {len(films)}")
        
        save_films_to_csv(films, OUTPUT_FILE_PATH)
        print(f"Film details successfully saved to {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")
