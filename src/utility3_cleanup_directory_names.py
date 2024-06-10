import os

clean_subdir_target = "keyframes_sentiments" # pick one in ["scenes", keyframes", "keyframes_sentiments"]

if clean_subdir_target == "scenes":
    ROOT_DIR = os.path.join("..", "data", "scenes") # "keyframes")
    SUFFIX_TO_DEL = '-mp4' # '.mp4'
elif clean_subdir_target == "keyframes":
    ROOT_DIR = os.path.join("..", "data", "keyframes")
    SUFFIX_TO_DEL = '.mp4'
elif clean_subdir_target == "keyframes_sentiments":
    ROOT_DIR = os.path.join("..", "data", "keyframes_sentiments")
    SUFFIX_TO_DEL = '-mp4'
else:
    print(f"ERROR: Invalid clean_subdir_target: {clean_subdir_target}")
    exit()

def rename_dirs_with_suffix(root_dir, suffix_to_del):
    # Traverse the directory tree starting from the root directory
    for current_dir, subdirs, files in os.walk(root_dir, topdown=False):
        for subdir in subdirs:
            # Check if the directory name ends with the specified suffix
            if subdir.endswith(suffix_to_del):
                old_path = os.path.join(current_dir, subdir)
                # Create the new directory name by removing the suffix
                new_subdir_name = subdir[:-len(suffix_to_del)]
                new_path = os.path.join(current_dir, new_subdir_name)
                
                # Check if the new path already exists to avoid conflicts
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                else:
                    # Handle potential conflicts by creating a unique name
                    counter = 1
                    unique_new_path = new_path
                    while os.path.exists(unique_new_path):
                        unique_new_path = f"{new_path}_{counter}"
                        counter += 1
                    os.rename(old_path, unique_new_path)
                    print(f"Renamed: {old_path} -> {unique_new_path}")

# Example usage:
# Define the root directory and the suffix to remove
rename_dirs_with_suffix(ROOT_DIR, SUFFIX_TO_DEL)
