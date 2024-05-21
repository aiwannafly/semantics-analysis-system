from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import TermMention, Term


class ReferenceResolver(ABC):

    @abstractmethod
    def __call__(self, terms: List[TermMention], text: str) -> List[Term]:
        raise NotImplemented()
