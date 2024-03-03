from abc import ABC, abstractmethod
from typing import List, Iterator

from semantics_analysis.entities import Relation, Term


class RelationExtractor(ABC):

    @abstractmethod
    def process(self, text: str, terms: List[Term]) -> Iterator[Relation]:
        pass
