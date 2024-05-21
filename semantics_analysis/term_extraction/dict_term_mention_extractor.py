import json
from typing import List

from pymorphy3 import MorphAnalyzer

from semantics_analysis.entities import TermMention
from semantics_analysis.phrase_extractor import PhraseExtractor
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor, PredictionsType


class DictTermExtractor(TermMentionExtractor):

    def __init__(self, dict_path: str):
        with open(dict_path, 'r', encoding='utf-8') as f:
            terms_by_class = json.load(f)

        self.class_by_term = {}

        for class_, terms in terms_by_class.items():
            for term in terms:
                self.class_by_term[term.lower()] = class_

        self.phrase_extractor = PhraseExtractor()

    def __call__(self, text: str, predictions: PredictionsType = None) -> List[TermMention]:
        potential_terms = [phrase for phrase in self.phrase_extractor(text).keys()]

        term_mentions = []

        morph = MorphAnalyzer(lang='ru')

        for term in potential_terms:
            value = term.lower()

            # TODO: add normalization for phrases
            if ' ' not in value:
                value = morph.parse(value)[0].normal_form

            class_ = self.class_by_term.get(value, None)

            start_pos = text.find(term)
            end_pos = start_pos + len(term)

            if class_:
                if class_ == 'Subject':
                    class_ = 'Object'

                term_mentions.append(TermMention(term, class_, end_pos, text, source='dict'))

        return term_mentions


def main():
    text = "Компьютерная лингвистика (КЛ) является довольно молодым разделом науки."

    classifier = DictTermExtractor(dict_path='metadata/terms_by_class.json')

    labeled_terms = classifier(text)

    print(labeled_terms)


if __name__ == '__main__':
    main()
