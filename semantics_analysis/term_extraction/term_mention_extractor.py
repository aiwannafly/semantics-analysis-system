from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Dict

from semantics_analysis.entities import TermMention


PredictionsType = TypeVar(
    'PredictionsType',
    List[Tuple[str, str, int, int, int]],
    Dict[TermMention, List[Tuple[str, float]]],
    None
)


class TermMentionExtractor(ABC):
    @abstractmethod
    def __call__(self, text: str, predictions: PredictionsType = None) -> List[TermMention]:
        raise NotImplemented()
