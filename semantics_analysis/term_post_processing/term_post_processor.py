from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import TermMention


class TermPostProcessor(ABC):
    @abstractmethod
    def __call__(self, terms: List[TermMention]) -> List[TermMention]:
        raise NotImplementedError('This class is abstract.')
