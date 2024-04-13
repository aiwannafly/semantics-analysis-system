from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from semantics_analysis.entities import Term


class TermExtractor(ABC):

    @abstractmethod
    def __call__(self, text: str,
                 predictions: Optional[List[Tuple[str, str, int, int, int]]] = None
                 ) -> List[Term]:
        raise NotImplemented()
