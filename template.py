import os

def create_directory_structure(base_dir):
    # Define directories to create
    directories = [
        "data/classification/train/tb_positive",
        "data/classification/train/tb_negative",
        "data/classification/val/tb_positive",
        "data/classification/val/tb_negative",
        "data/classification/test/tb_positive",
        "data/classification/test/tb_negative",
        "data/segmentation/images",
        "data/segmentation/masks",
        "data/processed",
        "models",
        "notebooks",
        "src/segmentation",
        "src/classification",
        "src/preprocessing",
        "src/api",
        "scripts",
        "tests"
    ]

    # Create directories
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create important files
    files = [
        "requirements.txt",
        "Dockerfile",
        ".gitignore",
        "README.md"
    ]

    for file_name in files:
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(f"# {file_name}\n")  # Add a simple header to the file
        print(f"Created file: {file_path}")


if __name__ == "__main__":
    # Set the base directory to the current directory (assuming you're running this inside the project folder)
    base_dir = os.getcwd()  # The current working directory, which is your existing project folder
    
    # Create the directory structure
    create_directory_structure(base_dir)
    print(f"Directory structure created successfully inside: {base_dir}")
