from abc import ABC, abstractmethod


class TermNormalizer(ABC):

    @abstractmethod
    def __call__(self, term: str) -> str:
        raise NotImplemented('This class is abstract.')
