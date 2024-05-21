from abc import ABC, abstractmethod
from typing import List, Iterator

from semantics_analysis.entities import TermMention


class TermNormalizer(ABC):
    @abstractmethod
    def __call__(self, term: str) -> str:
        raise NotImplemented('This class is abstract.')

    def normalize_all(self, term_mentions: List[TermMention]) -> Iterator[str]:
        for mention in term_mentions:
            mention.norm_value = self(mention.value)
            yield mention.norm_value
