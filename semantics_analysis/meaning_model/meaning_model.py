from abc import ABC, abstractmethod

from semantics_analysis.meaning_model.meaning import Meaning


class MeaningModel(ABC):

    @abstractmethod
    def get_meaning(self, text: str) -> Meaning:
        raise NotImplemented()
