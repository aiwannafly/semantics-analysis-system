from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import Term, ClassifiedTerm


class TermClassifier(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[Term]) -> List[ClassifiedTerm]:
        raise NotImplemented()
