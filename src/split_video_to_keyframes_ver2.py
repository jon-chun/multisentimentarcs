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

def create_output_subdir(base_output_path, video_filename):
    sanitized_filename = sanitize_filename(video_filename).replace(' ', '_')
    subdir_name = os.path.splitext(sanitized_filename)[0]
    output_subdir = os.path.join(base_output_path, subdir_name)
    try:
        os.makedirs(output_subdir, exist_ok=True)
        logging.info(f"Created output subdir: {output_subdir}")
        print(f"Created output subdir: {output_subdir}")
    except Exception as e:
        logging.error(f"Error creating output subdir {output_subdir}: {e}")
        print(f"Error creating output subdir {output_subdir}: {e}")
    return output_subdir

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

    command = [
        "scenedetect",
        "--input", input_file_path,
        "-s", stats_file_path,
        "detect-content",
        "list-scenes",
        "--output", scenes_file_path
    ]
    logging.info(f"Command to execute: {' '.join(command)}")
    print(f"Command to execute: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        logging.info(f"Processed {input_file_path} successfully. Output written to {output_subdir}")
        print(f"Processed {input_file_path} successfully. Output written to {output_subdir}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {input_file_path}: {e}")
        print(f"Error processing {input_file_path}: {e}")

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
                    
                    output_subdir = create_output_subdir(output_dir_base_path, filename)
                    process_video_file(input_file_path, output_subdir)
        except Exception as e:
            logging.error(f"Error processing directory {leaf_dir}: {e}")
            print(f"Error processing directory {leaf_dir}: {e}")

    logging.info("All videos processed.")
    print("All videos processed.")

if __name__ == "__main__":
    main()
