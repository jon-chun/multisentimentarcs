import os
import logging
import csv
import cv2
from datetime import datetime

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"keyframe_extraction_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

# Constants
RESET_FLAG = True  # Set to True to delete old keyframe images before writing new ones

def find_leaf_directories(base_path):
    leaf_dirs = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  # No subdirectories, it's a leaf
            leaf_dirs.append(root)
    logging.info(f"Leaf directories found: {leaf_dirs}")
    return leaf_dirs

"""
def extract_keyframes(base_output_path):
    for root, dirs, files in os.walk(base_output_path):
        for file in files:
            if file.endswith("-Scenes.csv"):
                process_scene_file(root, file)
""";

def extract_keyframes(base_output_path):
    for root, dirs, files in os.walk(base_output_path):
        # Remove "keyframes" from dirs to prevent os.walk from visiting it
        if "keyframes" in dirs:
            dirs.remove("keyframes")
        for file in files:
            if file.endswith("-Scenes.csv"):
                process_scene_file(root, file)


def process_scene_file(root, file):
    scene_file_path = os.path.join(root, file)
    keyframes_dir = os.path.join(root, "keyframes")
    if RESET_FLAG and os.path.exists(keyframes_dir):
        for filename in os.listdir(keyframes_dir):
            file_path = os.path.join(keyframes_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted old keyframe image: {file_path}")

    if not os.path.exists(keyframes_dir):
        os.makedirs(keyframes_dir)
        print(f"Created directory: {keyframes_dir}")
    
    print(f"Processing film: {file}")

    with open(scene_file_path, 'r') as csvfile:
        # Skip the first line (timecode list)
        next(csvfile)
        
        reader = csv.DictReader(csvfile)
        if not validate_csv_headers(reader.fieldnames):
            logging.error(f"CSV file {file} is missing required headers.")
            return
        
        for row in reader:
            try:
                start_time = float(row['Start Time (seconds)'])
                end_time = float(row['End Time (seconds)'])
                if start_time >= end_time:
                    logging.warning(f"Invalid scene times in {file}: start_time {start_time} >= end_time {end_time}")
                    continue
                scene_no = row['Scene Number']
                save_keyframe(root, keyframes_dir, scene_no, file, start_time, end_time)
            except ValueError as e:
                logging.error(f"Error processing scene row in {file}: {e}")
            except KeyError as e:
                logging.error(f"Missing expected column in {file}: {e}")

def validate_csv_headers(headers):
    # Updated required headers to match the actual CSV structure
    required_headers = {'Start Time (seconds)', 'End Time (seconds)', 'Scene Number'}
    if not all(header in headers for header in required_headers):
        print(f"CSV file is missing required headers: {required_headers - set(headers)}")
        return False
    return True

def save_keyframe(root, keyframes_dir, scene_no, scene_file, start_time, end_time):
    # Determine the genre from the directory structure
    genre = os.path.basename(os.path.dirname(root))
    
    # Construct the correct path to the video file
    video_filename = scene_file.replace("-Scenes.csv", ".mp4")
    video_path = os.path.join("..", "data", "videos", genre, video_filename)
    
    if not os.path.exists(video_path):
        logging.error(f"Video file {video_path} does not exist.")
        return

    best_keyframe = get_best_keyframe(video_path, start_time, end_time)
    
    if best_keyframe is not None:
        frame_filename = f"scene{scene_no}_{os.path.splitext(video_filename)[0]}.png"
        frame_path = os.path.join(keyframes_dir, frame_filename)
        cv2.imwrite(frame_path, best_keyframe)
        print(f"Saved keyframe for scene {scene_no} at {frame_path}")
        logging.info(f"Saved keyframe for scene {scene_no} at {frame_path}")
    else:
        logging.error(f"Failed to extract a keyframe for scene {scene_no} from {video_path}")

def get_best_keyframe(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logging.error(f"Failed to get FPS from video file {video_path}")
        cap.release()
        return None

    # Calculate the midpoint frame
    midpoint_time = (start_time + end_time) / 2
    midpoint_frame = int(midpoint_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame)
    success, midpoint_image = cap.read()

    if not success:
        logging.warning(f"Failed to capture midpoint frame from {video_path}. Trying the start frame.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        success, midpoint_image = cap.read()

    if not success:
        logging.error(f"Failed to capture any frame between {start_time} and {end_time} from {video_path}")
        cap.release()
        return None

    best_keyframe = midpoint_image
    best_score = calculate_image_score(midpoint_image)

    # Analyze frames between start and end to find a better keyframe
    frame_no = int(start_time * fps)
    end_frame = int(end_time * fps)
    while frame_no <= end_frame:
        success, frame = cap.read()
        if not success:
            break

        score = calculate_image_score(frame)
        if score > best_score:
            best_score = score
            best_keyframe = frame

        frame_no += 1

    cap.release()
    return best_keyframe

def calculate_image_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def main():
    base_output_path = os.path.join("..", "data", "keyframes")
    extract_keyframes(base_output_path)
    print("Keyframe extraction completed.")
    logging.info("Keyframe extraction completed.")

if __name__ == "__main__":
    main()
