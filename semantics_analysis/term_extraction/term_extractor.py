from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import Term


class TermExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str) -> List[Term]:
        raise NotImplemented()
