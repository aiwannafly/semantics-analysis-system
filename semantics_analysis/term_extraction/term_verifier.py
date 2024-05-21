from abc import abstractmethod, ABC
from typing import List, Optional, Iterator

from semantics_analysis.entities import TermMention


class TermVerifier(ABC):
    @abstractmethod
    def __call__(self, term_value: str, term_class: str, text: str) -> bool:
        raise NotImplemented('Abstract method.')

    def filter_terms(self, term_mentions: List[TermMention]) -> Iterator[Optional[TermMention]]:
        for mention in term_mentions:
            if self(mention.value, mention.class_, mention.text):
                yield mention
            else:
                yield None
