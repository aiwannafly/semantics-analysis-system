from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Tuple

from semantics_analysis.entities import Relation, ClassifiedTerm, GroupedTerm


class RelationExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[ClassifiedTerm]) -> Iterator[Relation]:
        pass

    @abstractmethod
    def get_pairs_to_consider(self, terms: List[ClassifiedTerm]) -> List[Tuple[ClassifiedTerm, ClassifiedTerm]]:
        pass

    @abstractmethod
    def analyze_term_pairs(
            self,
            text: str,
            term_pairs: List[Tuple[ClassifiedTerm, ClassifiedTerm]]
    ) -> Iterator[Optional[Relation]]:
        pass
