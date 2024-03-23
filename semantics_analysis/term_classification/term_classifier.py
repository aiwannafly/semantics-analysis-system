from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from semantics_analysis.entities import Term, ClassifiedTerm


class TermClassifier(ABC):

    @abstractmethod
    def __call__(self, text: str, terms: List[Term]) -> List[ClassifiedTerm]:
        raise NotImplemented()

    @abstractmethod
    def run_and_save_predictions(
            self,
            text: str,
            terms: List[Term],
            predictions: Dict[Term, List[Tuple[str, float]]]
    ) -> List[ClassifiedTerm]:
        raise NotImplemented()
