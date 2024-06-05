import os
import logging
import re
from datetime import datetime
import subprocess
import pandas as pd

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
SAVE_STATS = True
MAX_SCENE_COUNT = 1000 # 300
MIN_SCENE_COUNT = 100 # 10
MAX_RETRY_ATTEMPTS = 5
INITIAL_THRESHOLD = 30.0
THRESHOLD_ADJUSTMENT = 5.0
DESIRED_SCENE_COUNT_MIN = 15
DESIRED_SCENE_COUNT_MAX = 25
PARTIAL_VIDEO_PERCENTAGE = 10
MIN_SCENE_FILE_KB = 5
CSV_FILE_PATH = os.path.join("..", "data", "keyframes", "scene_threshold_and_count.csv")
HISTORY_FILE_PATH = os.path.join("..", "data", "keyframes", "scenedetect_history.csv")

# Initialize DataFrame
columns = ["filename", "size", "scene_count", "threshold"]
if os.path.exists(CSV_FILE_PATH):
    scene_df = pd.read_csv(CSV_FILE_PATH)
else:
    scene_df = pd.DataFrame(columns=columns)

if os.path.exists(HISTORY_FILE_PATH):
    history_df = pd.read_csv(HISTORY_FILE_PATH)
else:
    history_df = pd.DataFrame(columns=columns)

def sanitize_filename(filename):
    illegal_chars_pattern = re.compile(r'[<>:"/\\|?*.,;]')
    sanitized_name = illegal_chars_pattern.sub('-', filename)
    logging.info(f"Sanitized filename: {sanitized_name}")
    return sanitized_name

def find_leaf_directories(base_path):
    leaf_dirs = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # No subdirectories, it's a leaf
            leaf_dirs.append(root)
    logging.info(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs

def create_output_subdir(base_output_path, leaf_dir, video_filename):
    genre = os.path.basename(leaf_dir)
    sanitized_filename = sanitize_filename(video_filename).replace(' ', '_')
    subdir_name = os.path.splitext(sanitized_filename)[0]
    output_subdir = os.path.join(base_output_path, genre, subdir_name)
    try:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Created output subdir: {output_subdir}")
    except Exception as e:
        logging.error(f"Error creating output subdir {output_subdir}: {e}")
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
        return 0

def adjust_threshold(current_threshold, scene_count):
    if scene_count > DESIRED_SCENE_COUNT_MAX:
        return current_threshold + THRESHOLD_ADJUSTMENT
    elif scene_count < DESIRED_SCENE_COUNT_MIN:
        return max(0, current_threshold - THRESHOLD_ADJUSTMENT)
    return current_threshold

def partial_scene_detection(input_file_path, output_subdir, threshold, percentage):
    video_filename = os.path.basename(input_file_path)
    partial_scenes_file_path = os.path.join(output_subdir, f"{video_filename}_partial_scenes.csv")
    
    command = [
        "scenedetect",
        "--input", input_file_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_subdir,
        "--duration", f"{percentage}%"
    ]
    logging.info(f"Partial command to execute: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Partial scene detection for {input_file_path} completed. Output written to {output_subdir}")
        scene_count = count_scenes(partial_scenes_file_path)
        os.remove(partial_scenes_file_path)  # Clean up the partial scenes file
        return scene_count
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in partial scene detection for {input_file_path}: {e}")
        return 0

def get_previous_threshold(filename, current_threshold):
    previous_runs = scene_df[scene_df["filename"] == filename]
    if not previous_runs.empty:
        below_min = previous_runs[previous_runs["scene_count"] < MIN_SCENE_COUNT]
        above_max = previous_runs[previous_runs["scene_count"] > MAX_SCENE_COUNT]
        if not below_min.empty and above_max.empty:
            # All previous values of scene_count were < MIN_SCENE_COUNT
            new_threshold = 0.5 * below_min["threshold"].min()
            logging.info(f"All previous scene counts < MIN_SCENE_COUNT. Setting new threshold to {new_threshold}.")
            return new_threshold
        elif not above_max.empty and below_min.empty:
            # All previous values of scene_count were > MAX_SCENE_COUNT
            new_threshold = 1.25 * above_max["threshold"].max()
            logging.info(f"All previous scene counts > MAX_SCENE_COUNT. Setting new threshold to {new_threshold}.")
            return new_threshold
        elif not below_min.empty and not above_max.empty:
            # Previous values of scene_count were both < MIN_SCENE_COUNT and > MAX_SCENE_COUNT
            low_threshold = below_min["threshold"].max()
            high_threshold = above_max["threshold"].min()
            new_threshold = (low_threshold + high_threshold) / 2
            logging.info(f"Scene counts < MIN_SCENE_COUNT and > MAX_SCENE_COUNT. Setting new threshold to {new_threshold}.")
            return new_threshold
    return current_threshold

def should_skip_file(filename):
    previous_runs = history_df[history_df["filename"] == filename]
    if not previous_runs.empty:
        if all(previous_runs["scene_count"] < MIN_SCENE_COUNT) or all(previous_runs["scene_count"] > MAX_SCENE_COUNT):
            logging.info(f"Skipping {filename} based on previous runs.")
            return True
    return False

def process_video_file(input_file_path, output_subdir):
    video_filename = os.path.basename(input_file_path)
    logging.info(f"Processing video file: {video_filename}")
    
    video_basename = os.path.splitext(video_filename)[0]
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    logging.info(f"Scenes file path: {scenes_file_path}")

    # Check if the file should be skipped based on history and existing file size
    if should_skip_file(video_filename) and os.path.isfile(scenes_file_path) and os.path.getsize(scenes_file_path) > MIN_SCENE_FILE_KB * 1024:
        logging.info(f"File {video_filename} already processed and valid. Skipping.")
        return

    current_threshold = get_previous_threshold(video_filename, INITIAL_THRESHOLD)
    video_size = os.path.getsize(input_file_path)
    for attempt in range(MAX_RETRY_ATTEMPTS):
        # Perform partial scene detection
        partial_scene_count = partial_scene_detection(input_file_path, output_subdir, current_threshold, PARTIAL_VIDEO_PERCENTAGE)
        logging.info(f"Partial scene count for {video_filename}: {partial_scene_count}")

        if DESIRED_SCENE_COUNT_MIN <= partial_scene_count <= DESIRED_SCENE_COUNT_MAX:
            # If partial scene count is within desired range, proceed with full scene detection
            command = [
                "scenedetect",
                "--input", input_file_path,
                "detect-content", "--threshold", str(current_threshold),
                "list-scenes",
                "-o", output_subdir
            ]
            if SAVE_STATS:
                command.insert(4, f"-s {os.path.join(output_subdir, f'{video_basename}.stats-detectcontent.csv')}")

            logging.info(f"Command to execute: {' '.join(command)}")
            
            try:
                subprocess.run(command, check=True)
                logging.info(f"Processed {input_file_path} successfully. Output written to {output_subdir}")

                # Check the number of scenes detected
                scene_count = count_scenes(scenes_file_path)
                logging.info(f"Scene count for {video_filename}: {scene_count}")

                # Record the results in the DataFrame
                scene_df.loc[len(scene_df)] = [video_filename, video_size, scene_count, current_threshold]
                scene_df.to_csv(CSV_FILE_PATH, index=False)
                history_df.loc[len(history_df)] = [video_filename, video_size, scene_count, current_threshold]
                history_df.to_csv(HISTORY_FILE_PATH, index=False)
                
                if MIN_SCENE_COUNT <= scene_count <= MAX_SCENE_COUNT:
                    break  # Successful processing if scene count is within the desired range
            except subprocess.CalledProcessError as e:
                logging.error(f"Error processing {input_file_path}: {e}")

        # Adjust threshold and retry if the partial scene count is out of range
        current_threshold = adjust_threshold(current_threshold, partial_scene_count)
        logging.warning(f"Partial scene count ({partial_scene_count}) out of range. Adjusting threshold to {current_threshold} and retrying...")

        if attempt == MAX_RETRY_ATTEMPTS - 1:
            logging.error(f"Failed after {MAX_RETRY_ATTEMPTS} attempts for {video_filename}. Skipping...")

def main():
    input_dir_base_path = os.path.join("..", "data", "videos")
    logging.info(f"Input directory base path: {input_dir_base_path}")

    output_dir_base_path = os.path.join("..", "data", "keyframes")
    logging.info(f"Output directory base path: {output_dir_base_path}")

    try:
        leaf_dirs = find_leaf_directories(input_dir_base_path)
    except Exception as e:
        logging.error(f"Error finding leaf directories: {e}")
        return
    
    for leaf_dir in leaf_dirs:
        logging.info(f"Processing directory: {leaf_dir}")
        
        try:
            for filename in os.listdir(leaf_dir):
                if filename.endswith(".mp4"):
                    input_file_path = os.path.join(leaf_dir, filename)
                    logging.info(f"Found MP4 file: {input_file_path}")
                    
                    output_subdir = create_output_subdir(output_dir_base_path, leaf_dir, filename)
                    process_video_file(input_file_path, output_subdir)

                    # Remove the output subdir if empty
                    if not os.listdir(output_subdir):
                        os.rmdir(output_subdir)
                        logging.info(f"Removed empty directory: {output_subdir}")
        except Exception as e:
            logging.error(f"Error processing directory {leaf_dir}: {e}")

    logging.info("All videos processed.")

if __name__ == "__main__":
    main()
