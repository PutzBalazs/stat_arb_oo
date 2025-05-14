from abc import ABC, abstractmethod
from typing import List, Dict
from ..core.token import Token

class Dex(ABC):
    @abstractmethod
    def fetch_and_prepare_data(self) -> List[Token]:
        """Fetch and prepare token data from the DEX"""
        pass

    @abstractmethod
    def execute_swap(self, src_token: str, dst_token: str, amount: float, slippage: float = 1.0) -> str:
        """Execute a swap transaction"""
        pass

