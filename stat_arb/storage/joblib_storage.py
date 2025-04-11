import os
import json
from .storage_interface import DataStorage  # Import the DataStorage interface

class JolibDataStorage(DataStorage):  # Implementing DataStorage interface
    def __init__(self, base_path="data/"):
        self.base_path = base_path

    def _get_file_path(self, folder, file_name):
        # Ensure the folder exists
        folder_path = os.path.join(self.base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, f"{file_name}.json")

    def save(self, data, folder, file_name):
        file_path = self._get_file_path(folder, file_name)
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def read(self, folder, file_name):
        file_path = self._get_file_path(folder, file_name)
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading data: {e}")
            return None
