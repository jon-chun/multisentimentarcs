import os
import json
import pandas as pd
import argparse

# Constants
DEFAULT_INPUT_ROOT_DIR_TRANSCRIPTS_JSON = os.path.join("..", "data", "transcripts")
DEFAULT_OUTPUT_ROOT_TRANSCRIPTS_COMBINED_CSV = os.path.join("..", "data", "transcripts_combined")

# Error handling
def crawl_directories(root_dir):
    """
    Crawl the given root directory and return a list of JSON file paths.
    """
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.json'):
                json_files.append(os.path.join(dirpath, file))
    return json_files

def process_json_file(json_file, min_text_len, skip_directions_flag):
    """
    Process the JSON file and return the processed dialog_combo_dict.
    """
    dialog_combo_dict = []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading file {json_file}: {e}")
        return []
    
    for entry in data:
        text = entry.get('text', '')
        if skip_directions_flag and text.startswith('[') and text.endswith(']'):
            continue
        
        if len(text) < min_text_len:
            continue  # Handle short texts in another pass
        
        dialog_combo_dict.append({
            'text': text,
            'start': entry.get('start', 0),
            'duration': entry.get('duration', 0)
        })
    
    dialog_combo_dict = merge_short_texts(dialog_combo_dict, min_text_len)
    dialog_combo_dict = add_time_midpoints(dialog_combo_dict)
    
    return dialog_combo_dict

def merge_short_texts(dialog_combo_dict, min_text_len):
    """
    Merge short text entries in dialog_combo_dict until all entries meet min_text_len.
    """
    merged_dict = []
    temp_text = ""
    temp_start = None
    temp_duration = 0
    
    for entry in dialog_combo_dict:
        text = entry['text']
        
        if len(text) < min_text_len:
            temp_text += " " + text
            if temp_start is None:
                temp_start = entry['start']
            temp_duration += entry['duration']
        else:
            if temp_text:
                merged_dict.append({
                    'text': temp_text.strip(),
                    'start': temp_start,
                    'duration': temp_duration
                })
                temp_text = ""
                temp_start = None
                temp_duration = 0
            merged_dict.append(entry)
    
    if temp_text:
        merged_dict.append({
            'text': temp_text.strip(),
            'start': temp_start,
            'duration': temp_duration
        })
    
    return merged_dict

def add_time_midpoints(dialog_combo_dict):
    """
    Add 'time_midpoint' to each entry in dialog_combo_dict.
    """
    for entry in dialog_combo_dict:
        entry['time_midpoint'] = entry['start'] + (0.5 * entry['duration'])
    
    return dialog_combo_dict

def save_to_csv(dialog_combo_dict, output_file):
    """
    Save dialog_combo_dict to a CSV file.
    """
    df = pd.DataFrame(dialog_combo_dict)
    try:
        df.to_csv(output_file, index=False)
    except IOError as e:
        print(f"Error saving file {output_file}: {e}")

def extract_genre_from_path(file_path):
    """
    Extract the genre from the file path. Assumes genre is the first subdirectory under the root directory.
    """
    parts = file_path.split(os.sep)
    for i, part in enumerate(parts):
        if part == 'transcripts':
            return parts[i + 1]
    return 'unknown'

def main(input_root_dir, output_root_dir, min_text_len, skip_directions_flag):
    """
    Main function to process all JSON files in the given root directory.
    """
    json_files = crawl_directories(input_root_dir)
    
    for json_file in json_files:
        dialog_combo_dict = process_json_file(json_file, min_text_len, skip_directions_flag)
        
        if dialog_combo_dict:
            genre = extract_genre_from_path(json_file)
            # Derive the output file name from the input file name
            file_name = os.path.basename(json_file).replace('.json', '_clean_transcript.csv')
            output_file = os.path.join(output_root_dir, genre, file_name)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            save_to_csv(dialog_combo_dict, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process SRT JSON files to compute sentiments.")
    parser.add_argument('input_root_dir', type=str, nargs='?', default=DEFAULT_INPUT_ROOT_DIR_TRANSCRIPTS_JSON,
                        help="Root directory to crawl for JSON files.")
    parser.add_argument('output_root_dir', type=str, nargs='?', default=DEFAULT_OUTPUT_ROOT_TRANSCRIPTS_COMBINED_CSV,
                        help="Directory to save the output CSV files.")
    parser.add_argument('--min_text_len', type=int, default=10, 
                        help="Minimum text length for merging short entries.")
    parser.add_argument('--skip_directions_flag', type=bool, default=True, 
                        help="Flag to skip non-spoken directions indicated by enclosed brackets.")
    
    args = parser.parse_args()
    
    main(args.input_root_dir, args.output_root_dir, args.min_text_len, args.skip_directions_flag)
