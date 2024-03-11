from typing import List

from semantics_analysis.entities import ClassifiedTerm
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor


#  The postprocessor merges terms with the same class that stand right close to each other.
class MergeCloseTermPostProcessor(TermPostProcessor):

    def __call__(self, terms: List[ClassifiedTerm]) -> List[ClassifiedTerm]:
        if not terms:
            return []

        orig_count = len(terms)

        terms: List[ClassifiedTerm] = sorted(terms, key=lambda t: t.end_pos)

        processed_terms = []

        merged_prev_term = False

        for i in range(len(terms) - 1):
            if merged_prev_term:
                merged_prev_term = False
                continue

            term, next_term = terms[i], terms[i + 1]

            if term.class_ != next_term.class_:
                processed_terms.append(term)
                continue

            if term.end_pos == next_term.start_pos:
                processed_terms.append(ClassifiedTerm(
                    value=term.value + next_term.value,
                    end_pos=next_term.end_pos,
                    term_class=term.class_,
                    text=term.text
                ))
                merged_prev_term = True
            elif term.end_pos + 1 == next_term.start_pos:
                processed_terms.append(ClassifiedTerm(
                    value=term.value + ' ' + next_term.value,
                    end_pos=next_term.end_pos,
                    term_class=term.class_,
                    text=term.text
                ))
                merged_prev_term = True
            else:
                processed_terms.append(term)

        if not merged_prev_term:  # last term is not merged
            processed_terms.append(terms[-1])

        new_count = len(processed_terms)

        if new_count == orig_count:
            return processed_terms
        else:
            return self(processed_terms)
