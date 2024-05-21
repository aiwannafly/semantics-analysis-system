from typing import List

from semantics_analysis.entities import TermMention
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor, PredictionsType


class CombinedTermExtractor(TermMentionExtractor):

    def __init__(self, *term_mention_extractors: TermMentionExtractor):
        self.term_mention_extractors = term_mention_extractors

    def __call__(self, text: str, predictions: PredictionsType = None) -> List[TermMention]:
        term_mentions = []

        for term_mention_extractor in self.term_mention_extractors:
            if not term_mentions:
                term_mentions.extend(term_mention_extractor(text, predictions))
                continue

            predicted = term_mention_extractor(text, predictions)

            older_terms = [t for t in term_mentions]

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
                    term_mentions.append(term)

        return term_mentions
