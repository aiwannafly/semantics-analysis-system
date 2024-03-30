from abc import ABC, abstractmethod
from typing import List, Iterator, Optional

from rich.progress import Progress

from semantics_analysis.entities import Relation, ClassifiedTerm


class RelationExtractor(ABC):

    @abstractmethod
    def __call__(
            self,
            text: str,
            terms: List[ClassifiedTerm],
            progress: Progress,
            considered_class1: Optional[str] = None,
            considered_class2: Optional[str] = None
    ) -> Iterator[Relation]:
        pass
