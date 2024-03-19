from abc import ABC, abstractmethod
from typing import List, Iterator, Optional

from semantics_analysis.entities import Relation, ClassifiedTerm


class RelationExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[ClassifiedTerm]) -> Iterator[Relation]:
        pass

    @abstractmethod
    def get_pairs_to_consider(self, terms: List[ClassifiedTerm]) -> List:
        pass

    @abstractmethod
    def analyze_term_pairs(
            self,
            text: str,
            term_pairs: List
    ) -> Iterator[Optional[Relation]]:
        pass
