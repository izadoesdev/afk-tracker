#!/usr/bin/env python
import os
import sys
import urllib.request
import shutil

def download_file(url, filename):
    """Download a file from a URL to the specified filename."""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def download_models():
    """Download required Haar cascade models."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download haarcascade files from OpenCV's GitHub repository
    haarcascade_files = [
        ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml", 
         "models/haarcascade_frontalface_default.xml"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml", 
         "models/haarcascade_eye.xml"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml", 
         "models/haarcascade_eye_tree_eyeglasses.xml"),
        ("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml", 
         "models/haarcascade_frontalface_alt.xml")
    ]
    
    for url, filename in haarcascade_files:
        if os.path.exists(filename):
            print(f"Haarcascade file already exists at {filename}")
        else:
            download_file(url, filename)
            # Create symlink or copy to current directory
            basename = os.path.basename(filename)
            if not os.path.exists(basename):
                try:
                    if sys.platform == "win32":
                        shutil.copy(filename, basename)
                        print(f"Copied {filename} to current directory")
                    else:
                        os.symlink(filename, basename)
                        print(f"Created symlink to {filename} in current directory")
                except Exception as e:
                    print(f"Error creating link/copy in current directory: {e}")
    
    # Also check if the files exist in OpenCV's data directory
    try:
        import cv2
        cascades_dir = cv2.data.haarcascades
        print(f"\nOpenCV Haar cascades directory: {cascades_dir}")
        print("You can also use the built-in cascades by leaving the cascade path arguments empty.")
    except (ImportError, AttributeError):
        print("\nCouldn't find OpenCV's built-in Haar cascades. Using downloaded files.")
    
    print("\nModel download complete!")
    print("You can now run:")
    print("  python run.py        (basic detector)")
    print("  python run.py --debug   (debug menu with visualization)")

if __name__ == "__main__":
    download_models() 