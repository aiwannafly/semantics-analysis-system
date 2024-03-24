from abc import ABC, abstractmethod
from typing import List

from semantics_analysis.entities import ClassifiedTerm, GroupedTerm


class ReferenceResolver(ABC):

    @abstractmethod
    def __call__(self, terms: List[ClassifiedTerm], text: str) -> List[GroupedTerm]:
        raise NotImplemented()