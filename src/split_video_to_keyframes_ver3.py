import os
import logging
import re
from datetime import datetime
import subprocess

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"keyframe_extraction_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Define constants
OVERWRITE_CSV = False
MAX_SCENE_COUNT = 300
MIN_SCENE_COUNT = 10
MAX_RETRY_ATTEMPTS = 5
INITIAL_THRESHOLD = 30.0
THRESHOLD_ADJUSTMENT = 5.0

def sanitize_filename(filename):
    illegal_chars_pattern = re.compile(r'[<>:"/\\|?*.,;]')
    sanitized_name = illegal_chars_pattern.sub('-', filename)
    logging.info(f"Sanitized filename: {sanitized_name}")
    print(f"Sanitized filename: {sanitized_name}")
    return sanitized_name

def find_leaf_directories(base_path):
    leaf_dirs = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # No subdirectories, it's a leaf
            leaf_dirs.append(root)
    logging.info(f"Leaf directories found: {leaf_dirs}")
    print(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs

def create_output_subdir(base_output_path, leaf_dir, video_filename):
    genre = os.path.basename(leaf_dir)
    sanitized_filename = sanitize_filename(video_filename).replace(' ', '_')
    subdir_name = os.path.splitext(sanitized_filename)[0]
    output_subdir = os.path.join(base_output_path, genre, subdir_name)
    try:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Created output subdir: {output_subdir}")
        print(f"Created output subdir: {output_subdir}")
    except Exception as e:
        logging.error(f"Error creating output subdir {output_subdir}: {e}")
        print(f"Error creating output subdir {output_subdir}: {e}")
    return output_subdir

def check_existing_files(output_subdir, video_basename):
    stats_file_path = os.path.join(output_subdir, f"{video_basename}.stats-detectcontent.csv")
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    
    files_exist = os.path.isfile(stats_file_path) and os.path.isfile(scenes_file_path)
    files_non_empty = os.path.getsize(stats_file_path) > 0 and os.path.getsize(scenes_file_path) > 0 if files_exist else False

    return files_exist, files_non_empty

def count_scenes(scenes_file_path):
    try:
        with open(scenes_file_path, 'r') as file:
            lines = file.readlines()
            # Assuming the first line is the header
            return len(lines) - 1  # Subtract 1 for the header
    except Exception as e:
        logging.error(f"Error counting scenes in {scenes_file_path}: {e}")
        print(f"Error counting scenes in {scenes_file_path}: {e}")
        return 0

def adjust_threshold(current_threshold, scene_count):
    if scene_count > MAX_SCENE_COUNT:
        return current_threshold + THRESHOLD_ADJUSTMENT
    elif scene_count < MIN_SCENE_COUNT:
        return max(0, current_threshold - THRESHOLD_ADJUSTMENT)
    return current_threshold

def process_video_file(input_file_path, output_subdir):
    video_filename = os.path.basename(input_file_path)
    logging.info(f"Processing video file: {video_filename}")
    print(f"Processing video file: {video_filename}")
    
    video_basename = os.path.splitext(video_filename)[0]
    stats_file_path = os.path.join(output_subdir, f"{video_basename}.stats-detectcontent.csv")
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    logging.info(f"Stats file path: {stats_file_path}")
    print(f"Stats file path: {stats_file_path}")
    logging.info(f"Scenes file path: {scenes_file_path}")
    print(f"Scenes file path: {scenes_file_path}")

    # Check if files exist and are non-empty if OVERWRITE_CSV is False
    if not OVERWRITE_CSV:
        files_exist, files_non_empty = check_existing_files(output_subdir, video_basename)
        if files_exist and files_non_empty:
            logging.info(f"CSV files already exist and are non-empty for {video_filename}. Skipping processing.")
            print(f"CSV files already exist and are non-empty for {video_filename}. Skipping processing.")
            return

    current_threshold = INITIAL_THRESHOLD
    for attempt in range(MAX_RETRY_ATTEMPTS):
        command = [
            "scenedetect",
            "--input", input_file_path,
            "-s", stats_file_path,
            "detect-content", f"--threshold", str(current_threshold),
            "list-scenes",
            "-o", output_subdir
        ]
        logging.info(f"Command to execute: {' '.join(command)}")
        print(f"Command to execute: {' '.join(command)}")
        
        try:
            subprocess.run(command, check=True)
            logging.info(f"Processed {input_file_path} successfully. Output written to {output_subdir}")
            print(f"Processed {input_file_path} successfully. Output written to {output_subdir}")

            # Check the number of scenes detected
            scene_count = count_scenes(scenes_file_path)
            logging.info(f"Scene count for {video_filename}: {scene_count}")
            print(f"Scene count for {video_filename}: {scene_count}")
            
            if MIN_SCENE_COUNT <= scene_count <= MAX_SCENE_COUNT:
                break  # Successful processing if scene count is within the desired range
            else:
                current_threshold = adjust_threshold(current_threshold, scene_count)
                logging.warning(f"Scene count ({scene_count}) out of range. Adjusting threshold to {current_threshold} and retrying...")
                print(f"Scene count ({scene_count}) out of range. Adjusting threshold to {current_threshold} and retrying...")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing {input_file_path}: {e}")
            print(f"Error processing {input_file_path}: {e}")
            if attempt == MAX_RETRY_ATTEMPTS - 1:
                logging.error(f"Failed after {MAX_RETRY_ATTEMPTS} attempts for {video_filename}. Skipping...")
                print(f"Failed after {MAX_RETRY_ATTEMPTS} attempts for {video_filename}. Skipping...")

def main():
    input_dir_base_path = os.path.join("..", "data", "videos")
    logging.info(f"Input directory base path: {input_dir_base_path}")
    print(f"Input directory base path: {input_dir_base_path}")

    output_dir_base_path = os.path.join("..", "data", "keyframes")
    logging.info(f"Output directory base path: {output_dir_base_path}")
    print(f"Output directory base path: {output_dir_base_path}")

    try:
        leaf_dirs = find_leaf_directories(input_dir_base_path)
    except Exception as e:
        logging.error(f"Error finding leaf directories: {e}")
        print(f"Error finding leaf directories: {e}")
        return
    
    for leaf_dir in leaf_dirs:
        logging.info(f"Processing directory: {leaf_dir}")
        print(f"Processing directory: {leaf_dir}")
        
        try:
            for filename in os.listdir(leaf_dir):
                if filename.endswith(".mp4"):
                    input_file_path = os.path.join(leaf_dir, filename)
                    logging.info(f"Found MP4 file: {input_file_path}")
                    print(f"Found MP4 file: {input_file_path}")
                    
                    output_subdir = create_output_subdir(output_dir_base_path, leaf_dir, filename)
                    process_video_file(input_file_path, output_subdir)
        except Exception as e:
            logging.error(f"Error processing directory {leaf_dir}: {e}")
            print(f"Error processing directory {leaf_dir}: {e}")

    logging.info("All videos processed.")
    print("All videos processed.")

if __name__ == "__main__":
    main()
