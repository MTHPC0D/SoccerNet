import os
import shutil

def check_and_clear_directory(directory_path):
    # Check if the path exists and is a directory
    if not os.path.exists(directory_path):
        print(f"The path {directory_path} does not exist.")
        return

    if not os.path.isdir(directory_path):
        print(f"The path {directory_path} is not a directory.")
        return

    # List the content of the directory
    files = os.listdir(directory_path)

    # Check if the directory is empty
    if not files:
        print(f"The directory {directory_path} is already empty.")
    else:
        # Delete the content of the directory
        for file in files:
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}. Reason: {e}")
        print(f"The content of the directory {directory_path} has been deleted.")

# Example usage
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/murge")
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/labels")
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/images")