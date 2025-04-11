from abc import ABC, abstractmethod

class DataStorage(ABC):
    @abstractmethod
    def save(self, data, folder, file_name):
        """Saves data to a specified location."""
        pass
    
    @abstractmethod
    def read(self, folder, file_name):
        """Reads data from a specified location."""
        pass
