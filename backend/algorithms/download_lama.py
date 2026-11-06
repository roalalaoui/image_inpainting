
from algorithms.lama.lama import download_weights

if __name__ == "__main__":
    print("Downloading LaMa model...")

    path = download_weights()

    print("Download complete!")
    print("Checkpoint:", path)
