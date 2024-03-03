from abc import ABC, abstractmethod
from typing import List


class TermExtractor(ABC):

    @abstractmethod
    def process(self, text: str) -> List[str]:
        raise NotImplemented()
