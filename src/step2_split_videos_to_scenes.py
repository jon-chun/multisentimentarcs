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
SAVE_STATS = False # This is currently not implemented/DO NOT USE
MAX_RETRY_ATTEMPTS = 10
INITIAL_THRESHOLD = 30.0
THRESHOLD_ADJUSTMENT_UP = 1.25
THRESHOLD_ADJUSTMENT_DOWN = 0.50
MAX_SCENE_COUNT = 600
MIN_SCENE_COUNT = 300
PARTIAL_VIDEO_PERCENTAGE = 10
MIN_SCENE_FILE_KB = 2
CSV_FILE_PATH = os.path.join("..", "data", "scenes", "scene_threshold_and_count.csv")
HISTORY_FILE_PATH = os.path.join("..", "data", "scenes", "scenedetect_history.csv")

# Initialize DataFrame
columns = ["filename", "size", "scene_count", "threshold"]
scene_df = pd.read_csv(CSV_FILE_PATH) if os.path.exists(CSV_FILE_PATH) else pd.DataFrame(columns=columns)
history_df = pd.read_csv(HISTORY_FILE_PATH) if os.path.exists(HISTORY_FILE_PATH) else pd.DataFrame(columns=columns)

def sanitize_filename(filename):
    """Sanitize filenames to remove special characters."""
    return re.sub(r'[<>:"/\\|?*.,;]', '-', filename).replace(' ', '_')

def find_leaf_directories(base_path):
    """Find directories without subdirectories."""
    return [root for root, dirs, files in os.walk(base_path) if not dirs]

def create_output_subdir(base_output_path, leaf_dir, video_filename):
    """Create output subdirectory for the given video."""
    genre = os.path.basename(leaf_dir)
    sanitized_filename = sanitize_filename(video_filename)
    subdir_name = os.path.splitext(sanitized_filename)[0]
    output_subdir = os.path.join(base_output_path, genre, sanitize_filename(subdir_name))
    scenes_file_path = os.path.join(output_subdir, f"{subdir_name}-Scenes.csv")

    if os.path.exists(scenes_file_path) and os.path.getsize(scenes_file_path) > MIN_SCENE_FILE_KB * 1024:
        logging.info(f"create_output_subdir(): Output file {scenes_file_path} for {video_filename} already exists and is non-empty. Skipping.")
        return output_subdir, True

    try:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"create_output_subdir(): Created output subdir: {output_subdir}")
    except Exception as e:
        logging.error(f"create_output_subdir(): Error creating output subdir {output_subdir}: {e}")
        return None, False

    return output_subdir, False

def count_scenes(scenes_file_path):
    """Count the number of scenes in the given file."""
    try:
        with open(scenes_file_path, 'r') as file:
            return len(file.readlines()) - 1
    except Exception as e:
        logging.error(f"count_scenes(): Error counting scenes in {scenes_file_path}: {e}")
        return 0
    
def trim_video(input_file_path, output_file_path, percentage):
    """Trim the input video to the specified percentage."""
    duration = get_video_duration(input_file_path)
    trim_duration = duration * (percentage / 100)
    command = [
        "ffmpeg", "-y",
        "-i", input_file_path,
        "-t", str(trim_duration),
        "-c", "copy",
        output_file_path
    ]
    logging.info(f"trim_video(): Trimming video command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        logging.info(f"trim_video(): Trimmed video saved to {output_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"trim_video(): Error trimming video {input_file_path}: {e}")
        return False


def clean_up_temp_file(output_subdir, video_filename):
    """Clean up temporary partial video file."""
    temp_file_path = os.path.join(output_subdir, f"{video_filename}_partial.mp4")
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

def get_previous_threshold(filename, current_threshold):
    """Retrieve the previous threshold used for the video."""
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
    """Determine if a video file should be skipped based on previous runs."""
    previous_runs = history_df[history_df["filename"] == filename]
    return not previous_runs.empty and any((MIN_SCENE_COUNT <= previous_runs["scene_count"]) & (previous_runs["scene_count"] <= MAX_SCENE_COUNT))

def validate_scene_count(input_file_path, scenes_file_path):
    """Validate the detected scenes against the video duration."""
    try:
        video_duration = get_video_duration(input_file_path)
        with open(scenes_file_path, 'r') as file:
            for line in file.readlines()[1:]:
                _, _, end_time = line.strip().split(',')
                if convert_time_to_seconds(end_time) > video_duration:
                    logging.error(f"validate_scene_count(): Invalid scene end time {end_time} for {input_file_path}. Video duration is {video_duration} seconds.")
                    return False
        return True
    except Exception as e:
        logging.error(f"validate_scene_count(): Error validating scene count for {input_file_path}: {e}")
        return False

def get_video_duration(file_path):
    """Get the duration of the video."""
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)
    except Exception as e:
        logging.error(f"get_video_duration(): Error getting video duration for {file_path}: {e}")
        return 0

def convert_time_to_seconds(time_str):
    """Convert time string to seconds."""
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def process_video_file(input_file_path, base_output_path, leaf_dir):
    """Process a single video file."""
    video_filename = os.path.basename(input_file_path)
    logging.info(f"process_video_file():Processing video file: {video_filename}")

    output_subdir, skip = create_output_subdir(base_output_path, leaf_dir, video_filename)
    if skip:
        return

    video_basename = os.path.splitext(video_filename)[0]
    scenes_file_path = os.path.join(output_subdir, f"{video_basename}-Scenes.csv")
    logging.info(f"process_video_file(): Scenes file path: {scenes_file_path}")

    if should_skip_processing(video_filename, scenes_file_path):
        return

    current_threshold = get_previous_threshold(video_filename, INITIAL_THRESHOLD)
    process_video_with_retries(input_file_path, output_subdir, video_filename, video_basename, scenes_file_path, current_threshold)

def should_skip_processing(video_filename, scenes_file_path):
    """Determine if processing should be skipped for a video."""
    if os.path.isfile(scenes_file_path) and os.path.getsize(scenes_file_path) > MIN_SCENE_FILE_KB * 1024:
        logging.info(f"should_skip_processing(): Output file for {video_filename} already exists and is not empty. Skipping.")
        return True
    return should_skip_file(video_filename)


def process_video_with_retries(input_file_path, output_subdir, video_filename, video_basename, scenes_file_path, current_threshold):

    attempt = 0
    while attempt < MAX_RETRY_ATTEMPTS:
        partial_scene_count = partial_scene_detection(input_file_path, output_subdir, current_threshold, PARTIAL_VIDEO_PERCENTAGE)
        logging.info(f"process_video_with_retries(): Partial scene count for {video_filename}: {partial_scene_count}")
        extrapolated_scene_count = (100 / PARTIAL_VIDEO_PERCENTAGE) * partial_scene_count

        if MIN_SCENE_COUNT <= extrapolated_scene_count <= MAX_SCENE_COUNT:
            logging.info(f"process_video_with_retries(): Partial scene count is within desired range for {video_filename}. Proceeding with full video.")
            full_scene_detection(input_file_path, output_subdir, current_threshold, video_basename, scenes_file_path)
            clean_up_temp_file(output_subdir, video_filename)
            print("process_video_with_retries(): FINISHED PROCESSING", video_filename, "\n\n")
            break
        else:

            current_threshold_old = current_threshold
            adjust_flag=None
            if MIN_SCENE_COUNT > extrapolated_scene_count:
                # Lower threshold to find more scenes
                current_threshold = current_threshold_old * THRESHOLD_ADJUSTMENT_DOWN
            else:
                # Raise the threshold to find fewer scenes
                current_threshold = current_threshold_old * THRESHOLD_ADJUSTMENT_UP

            print(f"\n\n  Adjusted threshold from {current_threshold_old} to {current_threshold} for {video_filename}")
            logging.warning(f"process_video_with_retries(): Partial scene count ({partial_scene_count}) out of range. Adjusting threshold to {current_threshold} and retrying...")

        attempt += 1

    logging.error(f"process_video_with_retries(): Failed after {MAX_RETRY_ATTEMPTS} attempts for {video_filename}. Skipping...")
    clean_up_temp_file(output_subdir, video_filename)


def clean_up_temp_file(output_subdir, video_filename):
    """Clean up temporary files."""
    try:
        partial_video_path = os.path.join(output_subdir, f"{os.path.splitext(video_filename)[0]}_partial.mp4")
        if os.path.exists(partial_video_path):
            os.remove(partial_video_path)
            logging.info(f"clean_up_temp_file(): Deleted temporary file: {partial_video_path}")

        # Check for empty directories and remove
        video_dir = os.path.join(output_subdir, os.path.splitext(video_filename)[0])
        if os.path.isdir(video_dir) and not os.listdir(video_dir):
            os.rmdir(video_dir)
            logging.info(f"clean_up_temp_file(): Removed empty directory: {video_dir}")

    except Exception as e:
        logging.error(f"clean_up_temp_file(): Error cleaning up temporary files for {video_filename}: {e}")


def partial_scene_detection(input_file_path, output_subdir, threshold, percentage):

    video_filename = os.path.basename(input_file_path)
    video_filename_no_ext = os.path.splitext(video_filename)[0]
    partial_video_path = os.path.join(output_subdir, f"{video_filename_no_ext}_partial.mp4")
    
    if not trim_video(input_file_path, partial_video_path, percentage):
        return 0

    output_partial_scenes_file_path = os.path.join(output_subdir, f"{video_filename_no_ext}_partial-Scenes.csv")

    command = [
        "scenedetect",
        "--input", partial_video_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_subdir
    ]
    logging.info(f"partial_scene_detection(): Partial command to execute: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        logging.info(f"partial_scene_detection(): Partial scene detection for {input_file_path} completed. Output written to {output_partial_scenes_file_path}")
        scene_count = count_scenes(output_partial_scenes_file_path)
    except subprocess.CalledProcessError as e:
        logging.error(f"partial_scene_detection(): Error in partial scene detection for {input_file_path}: {e}")
        return 0
    finally:
        try:
            os.remove(partial_video_path)
            logging.info(f"partial_scene_detection(): Deleted temporary file: {partial_video_path}")
        except Exception as e:
            logging.error(f"partial_scene_detection(): Error deleting temporary file {partial_video_path}: {e}")

    return scene_count


"""
def full_scene_detection(input_file_path, output_subdir, threshold, video_basename, scenes_file_path):

    output_scenes_dir = os.path.join(output_subdir, video_basename)
    os.makedirs(output_scenes_dir, exist_ok=True)

    logging.info(f"full_scene_detection(): Processing {input_file_path} with threshold {threshold} to {output_scenes_dir}")

    command = [
        "scenedetect",
        "--input", input_file_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_scenes_dir
    ]
    if SAVE_STATS:
        stats_file_path = os.path.join(output_scenes_dir, f"{video_basename}.stats.csv")
        command.extend(["--save-stats", stats_file_path])

    logging.info(f"full_scene_detection(): Command to execute: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        logging.info(f"full_scene_detection(): Processed {input_file_path} successfully. Output written to {output_scenes_dir}")

        scenes_file_path = os.path.join(output_scenes_dir, f"{video_basename}-Scenes.csv")
        scene_count = count_scenes(scenes_file_path)
        logging.info(f"full_scene_detection(): Scene count for {input_file_path}: {scene_count}")

        if not validate_scene_count(input_file_path, scenes_file_path):
            logging.warning(f"full_scene_detection(): Invalid scenes detected for {input_file_path}. Adjust your threshold settings.")
            return

        video_size = os.path.getsize(input_file_path)
        scene_df.loc[len(scene_df)] = [input_file_path, video_size, scene_count, threshold]
        scene_df.to_csv(CSV_FILE_PATH, index=False)
        history_df.loc[len(history_df)] = [input_file_path, video_size, scene_count, threshold]
        history_df.to_csv(HISTORY_FILE_PATH, index=False)
    except subprocess.CalledProcessError as e:
        logging.error(f"full_scene_detection(): Error processing {input_file_path}: {e}")
""";


# Assuming scene_df, CSV_FILE_PATH, history_df, and HISTORY_FILE_PATH are defined somewhere above in the actual code.

def full_scene_detection(input_file_path, output_subdir, threshold, video_basename, scenes_file_path):

    output_scenes_dir = os.path.join(output_subdir, video_basename)
    os.makedirs(output_scenes_dir, exist_ok=True)

    output_scenes_fullpath = os.path.join(output_scenes_dir, video_basename, "_partial-Scenes.csv")
    logging.info(f"full_scene_detection(): Processing {input_file_path} with threshold {threshold} to {output_scenes_fullpath}")

    # Path to the scenes file
    scenes_file_path = os.path.join(output_scenes_dir, f"{video_basename}-Scenes.csv")

    # Delete the file if it exists
    if os.path.exists(scenes_file_path):
        os.remove(scenes_file_path)
        logging.info(f"full_scene_detection(): Deleting file {scenes_file_path} if it exists.")

    command = [
        "scenedetect",
        "--input", input_file_path,
        "detect-content", "--threshold", str(threshold),
        "list-scenes",
        "-o", output_scenes_fullpath
    ]
    if SAVE_STATS:
        stats_file_path = os.path.join(output_scenes_dir, f"{video_basename}.stats.csv")
        command.extend(["--save-stats", stats_file_path])

    logging.info(f"full_scene_detection(): Command to execute: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        logging.info(f"full_scene_detection(): Processed {input_file_path} successfully. Output written to {output_scenes_dir}")

        scene_count = count_scenes(scenes_file_path)
        logging.info(f"full_scene_detection(): Scene count for {input_file_path}: {scene_count}")

        if not validate_scene_count(input_file_path, scenes_file_path):
            logging.warning(f"full_scene_detection(): Invalid scenes detected for {input_file_path}. Adjust your threshold settings.")
            return

        video_size = os.path.getsize(input_file_path)
        scene_df.loc[len(scene_df)] = [input_file_path, video_size, scene_count, threshold]
        scene_df.to_csv(CSV_FILE_PATH, index=False)
        history_df.loc[len(history_df)] = [input_file_path, video_size, scene_count, threshold]
        history_df.to_csv(HISTORY_FILE_PATH, index=False)
    except subprocess.CalledProcessError as e:
        logging.error(f"full_scene_detection(): Error processing {input_file_path}: {e}")


def main():
    """Main function to process all video files."""
    input_dir_base_path = os.path.join("..", "data", "videos")
    output_dir_base_path = os.path.join("..", "data", "scenes")

    leaf_dirs = find_leaf_directories(input_dir_base_path) if not SELECTED_VIDEOS_LIST else \
        [os.path.join(input_dir_base_path, partial_path.strip("/")) for partial_path in SELECTED_VIDEOS_LIST]

    for leaf_dir in leaf_dirs:
        logging.info(f"main(): Processing directory: {leaf_dir}")
        filename_list = sorted(os.listdir(leaf_dir))

        for filename in filename_list:
            if filename.endswith(".mp4"):
                input_file_path = os.path.join(leaf_dir, filename)
                logging.info(f"main(): Found MP4 file: {input_file_path}")
                process_video_file(input_file_path, output_dir_base_path, leaf_dir)

                if not os.listdir(output_dir_base_path):
                    os.rmdir(output_dir_base_path)
                    logging.info(f"main(): Removed empty directory: {output_dir_base_path}")

    logging.info("main(): All videos processed.")

if __name__ == "__main__":
    main()
