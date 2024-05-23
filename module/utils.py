import os

def get_project_root() -> str:
    file_path = os.path.abspath(__file__)
    while os.path.basename(file_path) != "mlgroup1":
        file_path = os.path.dirname(file_path)
    return file_path