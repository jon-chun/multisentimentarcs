import os
import logging
import re
from datetime import datetime
import subprocess
import pandas as pd

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"scene_extraction_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Define constants
SELECTED_VIDEOS_LIST = []

OVERWRITE_CSV = False
SAVE_STATS = True
MAX_RETRY_ATTEMPTS = 5
INITIAL_THRESHOLD = 30.0
THRESHOLD_ADJUSTMENT_UP = 1.25
THRESHOLD_ADJUSTMENT_DOWN = 0.50
MAX_SCENE_COUNT = 1000
MIN_SCENE_COUNT = 100
PARTIAL_VIDEO_PERCENTAGE = 10
PARTIAL_MIN_SCENE_COUNT = MIN_SCENE_COUNT / (100 / PARTIAL_VIDEO_PERCENTAGE)
PARTIAL_MAX_SCENE_COUNT = MAX_SCENE_COUNT / (100 / PARTIAL_VIDEO_PERCENTAGE)
MIN_SCENE_FILE_KB = 2
CSV_FILE_PATH = os.path.join("..", "data", "scenes", "scene_threshold_and_count.csv")
HISTORY_FILE_PATH = os.path.join("..", "data", "scenes", "scenedetect_history.csv")

# Initialize DataFrame
columns = ["filename", "size", "scene_count", "threshold"]
scene_df = pd.read_csv(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else pd.DataFrame(columns=columns)
history_df = pd.read_csv(HISTORY_FILE_PATH) if os.path.exists(HISTORY_FILE_PATH) else pd.DataFrame(columns=columns)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*.,;]', '-', filename).replace(' ', '_')

def find_leaf_directories(base_path):
    return [root for root, dirs, files in os.walk(base_path) if not dirs]

def create_output_subdir(base_output_path, leaf_dir, video_filename):
    genre = os.path.basename(leaf_dir)
    sanitized_filename = sanitize_filename(video_filename)
    subdir_name = os.path.splitext(sanitized_filename)[0]
    output_subdir = os.path.join(base_output_path, genre, subdir_name)
    
    scenes_file_path = os.path.join(output_subdir, f"{subdir_name}-Scenes.csv")
    if os.path.exists(scenes_file_path) and os.path.getsize(scenes_file_path) > MIN_SCENE_FILE_KB * 1024:
        logging.info(f"Output file {scenes_file_path} for {video_filename} already exists and is non-empty. Skipping.")
        return output_subdir, True

    try:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Created output subdir: {output_subdir}")
    except Exception as e:
        logging.error(f"Error creating output subdir {output_subdir}: {e}")
        return None, False

    return output_subdir, False

def check_existing_files(output_subdir, video_basename):
    stats_file_path = os.path.join(output_subdir, f"{video_basename}.stats-detectcontent.csv")
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    if os.path.isfile(stats_file_path) and os.path.isfile(scenes_file_path):
        if os.path.getsize(stats_file_path) > 0 and os.path.getsize(scenes_file_path) > 0:
            return True, True
    return False, False

def count_scenes(scenes_file_path):
    try:
        with open(scenes_file_path, 'r') as file:
            return len(file.readlines()) - 1
    except Exception as e:
        logging.error(f"Error counting scenes in {scenes_file_path}: {e}")
        return 0

def adjust_threshold(current_threshold, scene_count):
    if scene_count > PARTIAL_MAX_SCENE_COUNT:
        return current_threshold * THRESHOLD_ADJUSTMENT_UP
    elif scene_count < PARTIAL_MIN_SCENE_COUNT:
        return max(0, current_threshold * THRESHOLD_ADJUSTMENT_DOWN)
    return current_threshold

def trim_video(input_file_path, output_file_path, percentage):
    duration = get_video_duration(input_file_path)
    trim_duration = duration * (percentage / 100)
    command = [
        "ffmpeg", "-y",
        "-i", input_file_path,
        "-t", str(trim_duration),
        "-c", "copy",
        output_file_path
    ]
    logging.info(f"Trimming video command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        logging.info(f"Trimmed video saved to {output_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error trimming video {input_file_path}: {e}")
        return False

def partial_scene_detection(input_file_path, output_subdir, threshold, percentage):
    video_filename = os.path.basename(input_file_path)
    partial_video_path = os.path.join(output_subdir, f"{video_filename}_partial.mp4")
    
    if not trim_video(input_file_path, partial_video_path, percentage):
        return 0
    
    partial_scenes_file_path = os.path.join(output_subdir, f"{video_filename}_partial-Scenes.csv")
    
    command = [
        "scenedetect",
        "--input", partial_video_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_subdir
    ]
    logging.info(f"Partial command to execute: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Partial scene detection for {input_file_path} completed. Output written to {output_subdir}")
        scene_count = count_scenes(partial_scenes_file_path)
        return scene_count
    except subprocess.CalledProcessError as e:
        logging.error(f"Error in partial scene detection for {input_file_path}: {e}")
        return 0

def clean_up_temp_file(output_subdir, video_filename):
    temp_file_path = os.path.join(output_subdir, f"{video_filename}_partial.mp4")
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

def get_previous_threshold(filename, current_threshold):
    previous_runs = scene_df[scene_df["filename"] == filename]
    if not previous_runs.empty:
        below_min = previous_runs[previous_runs["scene_count"] < MIN_SCENE_COUNT]
        above_max = previous_runs[previous_runs["scene_count"] > MAX_SCENE_COUNT]
        if not below_min.empty and above_max.empty:
            return current_threshold * THRESHOLD_ADJUSTMENT_DOWN
        elif not above_max.empty and below_min.empty:
            return current_threshold * THRESHOLD_ADJUSTMENT_UP
        elif not below_min.empty and not above_max.empty:
            low_threshold = below_min["threshold"].max()
            high_threshold = above_max["threshold"].min()
            return (low_threshold + high_threshold) / 2
    return current_threshold

def should_skip_file(filename):
    previous_runs = history_df[history_df["filename"] == filename]
    if not previous_runs.empty:
        if any((MIN_SCENE_COUNT <= previous_runs["scene_count"]) & (previous_runs["scene_count"] <= MAX_SCENE_COUNT)):
            logging.info(f"Skipping {filename} based on previous successful runs.")
            return True
    return False

def validate_scene_count(input_file_path, scenes_file_path):
    try:
        video_duration = get_video_duration(input_file_path)
        with open(scenes_file_path, 'r') as file:
            for line in file.readlines()[1:]:
                _, _, end_time = line.strip().split(',')
                if convert_time_to_seconds(end_time) > video_duration:
                    logging.error(f"Invalid scene end time {end_time} for {input_file_path}. Video duration is {video_duration} seconds.")
                    return False
        return True
    except Exception as e:
        logging.error(f"Error validating scene count for {input_file_path}: {e}")
        return False

def get_video_duration(file_path):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)
    except Exception as e:
        logging.error(f"Error getting video duration for {file_path}: {e}")
        return 0

def convert_time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def process_video_file(input_file_path, base_output_path, leaf_dir):
    video_filename = os.path.basename(input_file_path)
    logging.info(f"\n\nProcessing video file: {video_filename}")
    
    output_subdir, skip = create_output_subdir(base_output_path, leaf_dir, video_filename)
    if skip:
        return
    
    video_basename = os.path.splitext(video_filename)[0]
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    logging.info(f"Scenes file path: {scenes_file_path}")

    if should_skip_processing(video_filename, scenes_file_path):
        return

    current_threshold = get_previous_threshold(video_filename, INITIAL_THRESHOLD)
    process_video_with_retries(input_file_path, output_subdir, video_filename, video_basename, scenes_file_path, current_threshold)

def should_skip_processing(video_filename, scenes_file_path):
    if os.path.isfile(scenes_file_path):
        if os.path.getsize(scenes_file_path) > MIN_SCENE_FILE_KB / 1024:
            logging.info(f"Output file for {video_filename} already exists and is not empty. Skipping.")
            return True
    if should_skip_file(video_filename):
        logging.info(f"File {video_filename} already processed successfully in the past. Skipping.")
        return True
    return False

def process_video_with_retries(input_file_path, output_subdir, video_filename, video_basename, scenes_file_path, current_threshold):
    for attempt in range(MAX_RETRY_ATTEMPTS):
        partial_scene_count = partial_scene_detection(input_file_path, output_subdir, current_threshold, PARTIAL_VIDEO_PERCENTAGE)
        logging.info(f"Partial scene count for {video_filename}: {partial_scene_count}")
        extrapolated_scene_count = (100 / PARTIAL_VIDEO_PERCENTAGE) * partial_scene_count
        
        print(f"TESTING IF PARTIAL SCENE COUNT WITHIN BOUNDS")
        if (PARTIAL_MIN_SCENE_COUNT <= partial_scene_count) & (partial_scene_count <= PARTIAL_MAX_SCENE_COUNT):
            print(f"  [TRUE]: TESTING IF PARTIAL SCENE COUNT WITHIN BOUNDS")
            logging.info(f"Partial scene count is within desired range for {video_filename}. Proceeding with full video.")
            # Ensure full video processing happens here
            full_scene_detection(input_file_path, output_subdir, current_threshold, video_basename, scenes_file_path)
            clean_up_temp_file(output_subdir, video_filename)
            return  # Return here to ensure the process is complete.
        else:
            print(f"  [FALSE]: TESTING IF PARTIAL SCENE COUNT WITHIN BOUNDS")
            current_threshold = adjust_threshold(current_threshold, partial_scene_count)
            logging.warning(f"Partial scene count ({partial_scene_count}) out of range. Adjusting threshold to {current_threshold} and retrying...")
        
        print(f"TESTING IF ATTEMPTS >= MAX_RETRY_ATTEMPTS: {MAX_RETRY_ATTEMPTS}")
        if attempt >= MAX_RETRY_ATTEMPTS - 1:
            print(f"  [TRUE]: TOO MANY ATTEMPTS {attempt}")
            logging.error(f"Failed after {MAX_RETRY_ATTEMPTS} attempts for {video_filename}. Skipping...")
            clean_up_temp_file(output_subdir, video_filename)
            return


def full_scene_detection(input_file_path, output_subdir, threshold, video_basename, scenes_file_path):
    command = [
        "scenedetect",
        "--input", input_file_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_subdir
    ]
    if SAVE_STATS:
        stats_file_path = os.path.join(output_subdir, f"{video_basename}.stats.csv")
        command.extend(["--save-stats", stats_file_path])

    logging.info(f"Command to execute: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Processed {input_file_path} successfully. Output written to {output_subdir}")

        scene_count = count_scenes(scenes_file_path)
        logging.info(f"Scene count for {input_file_path}: {scene_count}")

        if not validate_scene_count(input_file_path, scenes_file_path):
            logging.warning(f"Invalid scenes detected for {input_file_path}. Adjust your threshold settings.")
            return

        video_size = os.path.getsize(input_file_path)
        scene_df.loc[len(scene_df)] = [input_file_path, video_size, scene_count, threshold]
        scene_df.to_csv(CSV_FILE_PATH, index=False)
        history_df.loc[len(history_df)] = [input_file_path, video_size, scene_count, threshold]
        history_df.to_csv(HISTORY_FILE_PATH, index=False)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR in full_scene_detect() trying to run command: {command}")
        logging.error(f"Error processing {input_file_path}: {e}")

def main():
    input_dir_base_path = os.path.join("..", "data", "videos")
    logging.info(f"Input directory base path: {input_dir_base_path}")

    # output_dir_base_path = os.path.join("..", "data", "scenes")
    output_dir_base_path = os.path.join("..", "data", "scenes")
    logging.info(f"Output directory base path: {output_dir_base_path}")

    try:
        if len(SELECTED_VIDEOS_LIST) > 0:
            leaf_dirs = find_leaf_directories(input_dir_base_path)
        else:
            leaf_dirs = [os.path.join(input_dir_base_path, partial_path.strip("/")) for partial_path in SELECTED_VIDEOS_LIST]
    except Exception as e:
        logging.error(f"Error finding leaf directories: {e}")
        return
    
    for leaf_dir in leaf_dirs:
        logging.info(f"Processing directory: {leaf_dir}")
        filename_list = sorted(os.listdir(leaf_dir))

        try:
            for filename in filename_list:
                if filename.endswith(".mp4"):
                    input_file_path = os.path.join(leaf_dir, filename)
                    logging.info(f"Found MP4 file: {input_file_path}")
                    
                    previous_runs = history_df[history_df["filename"] == filename]
                    logging.info(f"Previous runs for {filename}:\n{previous_runs}")

                    output_subdir, skip = create_output_subdir(output_dir_base_path, leaf_dir, filename)
                    if skip:
                        continue

                    process_video_file(input_file_path, output_subdir, leaf_dir)

                    if not os.listdir(output_subdir):
                        os.rmdir(output_subdir)
                        logging.info(f"Removed empty directory: {output_subdir}")
        except Exception as e:
            logging.error(f"Error processing directory {leaf_dir}: {e}")
            continue

    logging.info("All videos processed.")

if __name__ == "__main__":
    main()
