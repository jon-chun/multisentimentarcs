import os
import csv
import logging
import cv2
import pandas as pd

# Define the root directories
INPUT_ROOT_SCENES_DIR = "../data/scenes/"
INPUT_ROOT_VIDEOS_DIR = "../data/movies/"

OUTPUT_ROOT_KEYFRAMES_DIR = "../data/keyframes/"

TARGET_FILMS_LIST = []  # Example list of target films 

def get_filmnames_and_scenes():
    data = []

    for root, dirs, files in os.walk(INPUT_ROOT_SCENES_DIR):
        for file in files:
            if file.endswith("-Scenes.csv") and not file.endswith("_partial-Scenes.csv"):
                scenescsv_file_fullpath = os.path.join(root, file)
                
                # Extract genre and film_filename from the path
                path_parts = scenescsv_file_fullpath.split(os.sep)
                try:
                    genre_index = path_parts.index('scenes') + 1
                    genre = path_parts[genre_index]
                    film_name_index = genre_index + 1
                    film_filename = path_parts[film_name_index]
                    
                    # Convert film_filename from "*-mp4" to "*.mp4"
                    film_filename = film_filename.replace("-mp4", ".mp4")
                    
                    # Append the details to the data list
                    data.append({"film_filename": film_filename, "genre": genre, "scenescsv_file_fullpath": scenescsv_file_fullpath})
                except (ValueError, IndexError) as e:
                    print(f"Error processing path: {scenescsv_file_fullpath}. Error: {e}")
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    
    # Return the DataFrame
    return df

def get_videofile_fullpath(film_filename):
    for root, dirs, files in os.walk(INPUT_ROOT_VIDEOS_DIR):
        for file in files:
            if file == film_filename:
                return os.path.join(root, file)
    return None

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

def save_keyframe(video_path, keyframes_dir, scene_no, film_filename, start_time, end_time):
    film_base_name = os.path.splitext(film_filename)[0]
    safe_film_base = film_base_name.replace("-", "_").replace(" ", "_").lower()
    frame_filename = f"scene{scene_no}_{safe_film_base}.png"
    frame_path = os.path.join(keyframes_dir, frame_filename)
    
    if not os.path.exists(frame_path):
        best_keyframe = get_best_keyframe(video_path, start_time, end_time)
        if best_keyframe is not None:
            cv2.imwrite(frame_path, best_keyframe)
            logging.info(f"Saved keyframe for scene {scene_no} at {frame_path}")
        else:
            logging.error(f"Failed to extract a keyframe for scene {scene_no} from {video_path}")
    else:
        logging.info(f"Keyframe for scene {scene_no} already exists at {frame_path}. Skipping.")

def process_scene_file(csv_file_path, video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory: {output_dir}")

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if not validate_csv_headers(reader.fieldnames):
            logging.error(f"CSV file {csv_file_path} is missing required headers.")
            return

        for row_index, row in enumerate(reader):
            try:
                start_time = float(row['Start Time (seconds)'])
                end_time = float(row['End Time (seconds)'])
                if start_time >= end_time:
                    logging.warning(f"Invalid scene times in {csv_file_path}: start_time {start_time} >= end_time {end_time}")
                    continue
                scene_no = row['Scene Number']
                save_keyframe(video_path, output_dir, scene_no, os.path.basename(csv_file_path), start_time, end_time)
            except ValueError as e:
                logging.error(f"Error processing scene row in {csv_file_path}: {e}")
            except KeyError as e:
                logging.error(f"Missing expected column in {csv_file_path}: {e}")

def extract_keyframes(df, base_output_path, reset_flag):
    for _, row in df.iterrows():
        film_filename = row['film_filename']
        genre = row['genre']
        scenescsv_file_fullpath = row['scenescsv_file_fullpath']
        videofile_fullpath = row['videofile_fullpath']
        
        if not videofile_fullpath:
            logging.warning(f"Video file for {film_filename} not found. Skipping.")
            continue
        
        output_dir = os.path.join(base_output_path, genre, os.path.splitext(film_filename)[0])
        
        if reset_flag and os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        process_scene_file(scenescsv_file_fullpath, videofile_fullpath, output_dir)

def main():
    # Get the DataFrame from the scenes directory
    df = get_filmnames_and_scenes()

    # Add videofile_fullpath to the DataFrame
    df['videofile_fullpath'] = df['film_filename'].apply(get_videofile_fullpath)
    
    # Save the DataFrame to a CSV file for reference
    df.to_csv("oswalk_data.csv", index=False)

    # Extract keyframes using the updated DataFrame
    # base_output_path = os.path.join("..", "data", "keyframes")
    reset_flag = True
    extract_keyframes(df, OUTPUT_ROOT_KEYFRAMES_DIR, reset_flag)
    logging.info("Keyframe extraction completed.")
    print("Keyframe extraction completed.")

if __name__ == "__main__":
    main()
