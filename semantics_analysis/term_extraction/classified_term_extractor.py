from abc import ABC, abstractmethod
from typing import List

from rich.progress import Progress

from semantics_analysis.entities import ClassifiedTerm


class ClassifiedTermExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str, progress: Progress) -> List[ClassifiedTerm]:
        raise NotImplemented()
