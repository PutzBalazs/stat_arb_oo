import os
import joblib
from .storage_interface import DataStorage

class JoblibDataStorage(DataStorage):
    def __init__(self, base_path="data/"):
        self.base_path = base_path

    def _get_file_path(self, folder, file_name):
        # Ensure the folder exists
        folder_path = os.path.join(self.base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, f"{file_name}.joblib")  # Use .joblib extension

    def save(self, data, folder, file_name):
        file_path = self._get_file_path(folder, file_name)
        try:
            joblib.dump(data, file_path)  # Use joblib.dump instead of pickle.dump
            print(f"Data saved to {file_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def read(self, folder, file_name):
        file_path = self._get_file_path(folder, file_name)
        try:
            return joblib.load(file_path)  # Use joblib.load instead of pickle.load
        except Exception as e:
            print(f"Error reading data: {e}")
            return None 