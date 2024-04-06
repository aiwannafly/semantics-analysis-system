from typing import List, Dict, Tuple

from semantics_analysis.entities import Term, ClassifiedTerm
from semantics_analysis.term_classification.term_classifier import TermClassifier


class HybridTermClassifier(TermClassifier):

    def __init__(self, *classifiers: TermClassifier):
        self.classifiers = classifiers

    def __call__(
            self,
            text: str,
            terms: List[Term],
            predictions: Dict[Term, List[Tuple[str, float]]]
    ) -> List[ClassifiedTerm]:

        classified_terms = []

        for classifier in self.classifiers:
            if not classified_terms:
                classified_terms.extend(classifier(text, terms, predictions))
                continue

            predicted = classifier(text, terms, predictions)

            older_terms = [t for t in classified_terms]

            for term in predicted:
                intersects = False

                for other in older_terms:
                    if term.start_pos <= other.start_pos:
                        term1, term2 = term, other
                    else:
                        term1, term2 = other, term

                    if term1.end_pos > term2.start_pos:
                        intersects = True
                        break

                if not intersects:
                    classified_terms.append(term)

        return classified_terms

