from abc import ABC, abstractmethod
from typing import List, Optional

from semantics_analysis.entities import Relation, Term, BoundedIterator


class RelationExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[Term]) -> BoundedIterator[Optional[Relation]]:
        pass
