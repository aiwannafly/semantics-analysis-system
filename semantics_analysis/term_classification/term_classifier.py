from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import Term


class TermClassifier(ABC):

    @abstractmethod
    def process(self, text: str, terms: List[str]) -> List[Term]:
        raise NotImplemented()
