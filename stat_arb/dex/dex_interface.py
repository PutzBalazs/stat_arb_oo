from abc import ABC, abstractmethod

class Dex(ABC):
    @abstractmethod
    def execute_swap(self, token_in: str, token_out: str, amount: float):
        pass

    @abstractmethod
    def get_data(self, token_address: str):
        pass
