import os
import subprocess
import glob

# STEP #1: Run these beforehand
# mkdir temp_frames
# ffmpeg -i sabrina_1954.mp4 -vf "select='gt(scene,0.4)'" -vsync vfr temp_frames/frame%04d.png


import os
import subprocess
import glob

# Directory containing the extracted frames
frames_dir = "temp_frames"

# Output file
output_file = "sabrina_1954.txt"

# Get a list of all PNG files in the frames directory
frame_files = glob.glob(os.path.join(frames_dir, "frame*.png"))

# Ensure we have frame files
if not frame_files:
    print("No frames found in the directory. Please check frame extraction step.")
else:
    print(f"Found {len(frame_files)} frame files.")

# Open the output file
with open(output_file, "w") as outfile:
    for frame_file in frame_files:
        # Run the ffprobe command for each frame
        cmd = [
            "ffprobe", 
            "-show_entries", "frame", 
            "-of", "compact=p=0:nk=1", 
            frame_file
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Debug print the command and output
        print(f"Processing file: {frame_file}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {result.returncode}")

        # Check if the command ran successfully
        if result.returncode == 0:
            # Filter and write the output to the file
            for line in result.stdout.splitlines():
                if 'media_type=video' in line:
                    outfile.write(line + "\n")
                    print(f"Writing line: {line}")
        else:
            # Print error message if ffprobe fails
            print(f"Error processing {frame_file}: {result.stderr}")

print(f"Metadata has been written to {output_file}")
