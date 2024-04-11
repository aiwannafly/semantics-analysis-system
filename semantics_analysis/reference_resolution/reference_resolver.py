from abc import ABC, abstractmethod
from typing import List

from rich.progress import Progress

from semantics_analysis.entities import ClassifiedTerm, GroupedTerm


class ReferenceResolver(ABC):

    @abstractmethod
    def __call__(self, terms: List[ClassifiedTerm], text: str, progress: Progress, normalize: bool = True) -> List[GroupedTerm]:
        raise NotImplemented()
