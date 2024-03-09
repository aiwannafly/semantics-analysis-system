from abc import ABC, abstractmethod
from typing import List, Iterator

from semantics_analysis.entities import Relation, ClassifiedTerm


class RelationExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[ClassifiedTerm]) -> Iterator[Relation]:
        pass
