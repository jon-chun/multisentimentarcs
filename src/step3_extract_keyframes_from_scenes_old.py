import os
import logging
import csv
import cv2
from datetime import datetime

# Inputs
videos_input_dir = "../data/videos/batch/"
scene_detect_csv_dir = "../data/keyframes/batch/"
scene_clips_dir = "../data/stills/"

TARGET_FILMS_LIST = [
    "Notorious_1946-Scenes.csv",
    "Rawhide_1951-Scenes.csv",
    "Royal_Wedding_1951-Scenes.csv",
    "The_Mob_1951-Scenes.csv",
]

# Set up logging to both terminal and the generated log file
current_time = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join("..", "data", f"keyframe_extraction_{current_time}.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])

def sanitize_directory_name(film_name):
    base_name = os.path.splitext(film_name)[0]
    safe_name = base_name.replace("-", "_").replace(" ", "_").lower()
    return safe_name

def delete_old_keyframes(directory):
    logging.info(f"Deleting old keyframe images in {directory}")
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if file.endswith(".png"):
            try:
                os.remove(file_path)
                logging.info(f"Deleted old keyframe image: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

def validate_csv_headers(headers):
    required_headers = {'Start Time (seconds)', 'End Time (seconds)', 'Scene Number'}
    missing_headers = required_headers - set(headers)
    if missing_headers:
        logging.error(f"CSV file is missing required headers: {missing_headers}")
        return False
    return True

def calculate_image_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

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

def save_keyframe(video_path, keyframes_dir, scene_no, scene_file, start_time, end_time):
    print(f"ENTERED save_keyframe()")
    film_base_name = os.path.splitext(scene_file.replace('-Scenes.csv', ''))[0]
    safe_film_path = film_base_name.replace("-", "_").replace(" ", "_").lower()
    safe_film_base = os.path.basename(safe_film_path)
    frame_filename = f"scene{scene_no}_{safe_film_base}.png"
    frame_path = os.path.join(keyframes_dir, frame_filename)
    
    print(f"    film_base_name: {film_base_name}")
    print(f"    safe_film_base: {safe_film_base}")
    print(f"    safe_film_path: {safe_film_path}")
    print(f"    frame_filename: {frame_filename}")
    print(f"        frame_path: {frame_path}")
    print(f"        start_time: {start_time}")
    print(f"          end_time: {end_time}")
    
    if not os.path.exists(frame_path):
        print(f' not os.path.exists(frame_path)')
        best_keyframe = get_best_keyframe(video_path, start_time, end_time)
        if best_keyframe is not None:
            print(f"writing best keyframe to frame_path: {frame_path}")
            cv2.imwrite(frame_path, best_keyframe)
            logging.info(f"Saved keyframe for scene {scene_no} at {frame_path}")
        else:
            logging.error(f"Failed to extract a keyframe for scene {scene_no} from {video_path}")
    else:
        print(f' os.path.exists(frame_path) so keyframe exists - skip   ')
        logging.info(f"Keyframe for scene {scene_no} already exists at {frame_path}. Skipping.")

def process_scene_file(csv_file_path, output_dir):
    logging.info(f"\n\nPROCESSING scene file: csv_file_path: {csv_file_path} with output_dir: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    with open(csv_file_path, 'r') as csvfile:
        print(f"   PROCESSING csv_file_path: {csv_file_path}")
        next(csvfile)  # Skip the first line (timecode list)
        reader = csv.DictReader(csvfile)
        if not validate_csv_headers(reader.fieldnames):
            logging.error(f"CSV file {csv_file_path} is missing required headers.")
            return

        for row_index, row in enumerate(reader):
            try:
                print(f"    PROCESSING row_index: {row_index}: {row.keys()}")
                start_time = float(row['Start Time (seconds)'])
                end_time = float(row['End Time (seconds)'])
                if start_time >= end_time:
                    logging.warning(f"Invalid scene times in {csv_file_path}: start_time {start_time} >= end_time {end_time}")
                    continue
                scene_no = row['Scene Number']
                movie_base_name = os.path.basename(csv_file_path)
                movie_name = movie_base_name.split('-Scenes')[0] + ".mp4"
                # csv_filename = os.path.splitext(os.path.basename(csv_file_path))[0] + ".mp4"
                print(f"  movie_name: {movie_name}")
                movie_path = os.path.join(videos_input_dir, movie_name)
                print(f"  movie_path: {movie_path}")
                
                if not os.path.exists(movie_path):
                    logging.error(f"Movie file {movie_path} does not exist.")
                    continue

                save_keyframe(movie_path, output_dir, scene_no, csv_file_path, start_time, end_time)
            except ValueError as e:
                logging.error(f"Error processing scene row in {csv_file_path}: {e}")
            except KeyError as e:
                logging.error(f"Missing expected column in {csv_file_path}: {e}")

def extract_keyframes(base_output_path, target_films_list, reset_flag):
    logging.info("Processing target films list.")
    for target_film in target_films_list:
        output_dir = os.path.join(base_output_path, sanitize_directory_name(target_film))
        csv_file_path = os.path.join(scene_detect_csv_dir, target_film)
        
        if os.path.exists(csv_file_path):
            if reset_flag and os.path.exists(output_dir):
                delete_old_keyframes(output_dir)
            process_scene_file(csv_file_path, output_dir)
        else:
            logging.warning(f"Scene file {csv_file_path} does not exist. Skipping.")

def main():
    base_output_path = os.path.join("..", "data", "stills")
    target_films_list = TARGET_FILMS_LIST
    reset_flag = True
    extract_keyframes(base_output_path, target_films_list, reset_flag)
    logging.info("Keyframe extraction completed.")
    print("Keyframe extraction completed.")

if __name__ == "__main__":
    main()
