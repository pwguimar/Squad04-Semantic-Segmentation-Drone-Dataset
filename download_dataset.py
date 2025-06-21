import kagglehub

# Download latest version
path = kagglehub.dataset_download("santurini/semantic-segmentation-drone-dataset")

print("Path to dataset files:", path)