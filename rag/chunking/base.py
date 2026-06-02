from abc import ABC, abstractmethod
from typing import List, Dict


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[Dict]:
        """
        Return list of chunks with at least:
        - text
        - start_char
        - end_char
        """
        pass