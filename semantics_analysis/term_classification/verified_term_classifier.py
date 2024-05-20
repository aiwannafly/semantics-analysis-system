from typing import List, Dict, Tuple

from rich.progress import Progress

from semantics_analysis.entities import Term, ClassifiedTerm
from semantics_analysis.term_classification.term_classifier import TermClassifier
from semantics_analysis.term_extraction.meaning_llm_term_extractor import MeaningLLMTermExtractor


class VerifiedTermClassifier(TermClassifier):

    def __init__(self, base_classifier: TermClassifier, progress: Progress):
        self.classifier = base_classifier
        self.progress = progress
        self.meaning_llm_term_extractor = MeaningLLMTermExtractor()

    def __call__(
            self,
            text: str,
            terms: List[Term],
            predictions: Dict[Term, List[Tuple[str, float]]]
    ) -> List[ClassifiedTerm]:

        classified_terms = self.classifier(text, terms, predictions=predictions)

        total = len(classified_terms)
        verify_task = self.progress.add_task(description=f'Term verification 0/{total}', total=total)

        verified_terms = []

        for t_idx, term in enumerate(classified_terms):
            verified = self.meaning_llm_term_extractor.verify_term(term.text, term.value, term.class_)

            self.progress.update(verify_task, description=f'Term verification {t_idx + 1}/{total}', advance=1)
            if verified:
                verified_terms.append(term)

        self.progress.remove_task(verify_task)

        return classified_terms

