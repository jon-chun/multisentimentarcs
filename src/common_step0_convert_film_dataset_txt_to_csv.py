import os
import re
import pandas as pd

# Define directory and file paths
dir_path = os.path.join("..", "data")

input_file = "dataset_yt_plain_small.txt"
output_file = "dataset_film_small_details.csv"

input_file_path = os.path.join(dir_path, input_file)
output_file_path = os.path.join(dir_path, output_file)

def sanitize_filename(filename):
    # Replace illegal characters with a hyphen
    sanitized_filename = re.sub(r'[\\/:*?"<>|]', '-', filename)
    # Remove leading and trailing periods
    sanitized_filename = re.sub(r'^\.+|\.+$', '', sanitized_filename)
    return sanitized_filename

def parse_film_data(text):
    # Define the regex patterns to extract the film details
    film_pattern = re.compile(
        r'film_name="(?P<name>[^"]+)"\n'
        r'film_year=(?P<year>\d+)\n'
        r'film_genre="(?P<genre>[^"]+)"\n'
        r'film_url="(?P<url>[^"]+)"'
    )
    xd
    
    films = []
    
    for match in film_pattern.finditer(text):
        film = match.groupdict()
        # Extract the YouTube ID from the film_url
        film_id_match = re.search(r'v=([a-zA-Z0-9_-]+)', film['url'])
        if film_id_match:
            film['video_id'] = film_id_match.group(1)
        else:
            film['video_id'] = None
        # Sanitize the film name for filename usage
        film['sanitized_name'] = sanitize_filename(film['name'])
        films.append(film)
        print(f"Parsed film: {film}")  # Debugging statement
    
    return films

def save_films_to_csv(films, output_file):
    if not films:
        print("No films to save")  # Debugging statement
        return
    
    # Create a DataFrame from the list of films
    df = pd.DataFrame(films)
    print(f"DataFrame to be saved:\n{df}")  # Debugging statement
    
    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved film details to {output_file}")

# Main function to execute the script
if __name__ == "__main__":
    # Read the text file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Parse the film data
    films = parse_film_data(text)
    print(f"Total films parsed: {len(films)}")  # Debugging statement
    
    # Save the films to a CSV file
    save_films_to_csv(films, output_file_path)
    print(f"Film details successfully saved to {output_file_path}")
