import json
from typing import List, Tuple, Dict

import spacy
from pymorphy3 import MorphAnalyzer

from semantics_analysis.entities import Term, ClassifiedTerm
from semantics_analysis.term_classification.term_classifier import TermClassifier


CONSIDERED_POS = ['NOUN', 'ADJ', 'PROPN', 'X']


class DictTermClassifier(TermClassifier):

    def __init__(self, dict_path: str):
        with open(dict_path, 'r', encoding='utf-8') as f:
            terms_by_class = json.load(f)

        self.class_by_term = {}

        for class_, terms in terms_by_class.items():
            for term in terms:
                self.class_by_term[term.lower()] = class_

        self.nlp = spacy.load("ru_core_news_sm")
        self.nlp.disable_pipes(["parser", "attribute_ruler", "lemmatizer"])

    def __call__(
            self,
            text: str,
            terms: List[Term],
            predictions: Dict[Term, List[Tuple[str, float]]]
    ) -> List[ClassifiedTerm]:
        potential_terms = self._extract_potential_terms(text)

        classified_terms = []

        morph = MorphAnalyzer(lang='ru')

        for term in potential_terms:
            value = term.value.lower()

            # TODO: add normalization for phrases
            if ' ' not in value:
                value = morph.parse(value)[0].normal_form

            class_ = self.class_by_term.get(value, None)

            if class_:
                classified_terms.append(ClassifiedTerm.from_term(class_, term))

        return classified_terms

    def _extract_potential_terms(self, text: str) -> List[Term]:
        doc = self.nlp(text)

        found_phrases = []

        curr_text = ''
        curr_term = ''

        adj_only = True
        for token in doc:
            remain_text = text[len(curr_text):]

            if token.pos_ in CONSIDERED_POS:
                is_under_term = True

                if token.pos_ != 'ADJ':
                    adj_only = False
            else:
                is_under_term = False

                if curr_term:
                    if not adj_only:
                        found_phrases.append(Term(value=curr_term.strip(), text=text, end_pos=len(curr_text)))

                    curr_term = ''
                    adj_only = True

            for i in range(len(remain_text)):
                curr_text += remain_text[i]

                if is_under_term:
                    curr_term += remain_text[i]

                if curr_text.endswith(token.text):
                    break

        if curr_term and not adj_only:
            found_phrases.append(Term(value=curr_term.strip(), text=text, end_pos=len(curr_text)))

        return found_phrases


def main():
    text = "Компьютерная лингвистика (КЛ) является довольно молодым разделом науки."

    classifier = DictTermClassifier(dict_path='metadata/terms_by_class.json')

    labeled_terms = classifier(text, [], {})

    print(labeled_terms)


if __name__ == '__main__':
    main()
