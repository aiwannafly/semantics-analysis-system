from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import ClassifiedTerm


class TermPostProcessor(ABC):
    @abstractmethod
    def __call__(self, terms: List[ClassifiedTerm]) -> List[ClassifiedTerm]:
        raise NotImplementedError('This class is abstract.')
