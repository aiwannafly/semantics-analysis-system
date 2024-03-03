from typing import List

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch import argmax, inference_mode

from semantics_analysis.term_extraction.term_extractor import TermExtractor

LABEL_LIST = ['O', 'B-TERM', 'I-TERM']


def tag2int(tag: str) -> int:
    if tag not in LABEL_LIST:
        if tag.startswith('B'):
            return 1
        elif tag.startswith('I'):
            return 2
        return 0

    return LABEL_LIST.index(tag)


def int2tag(idx: int) -> str:
    return LABEL_LIST[idx]


def normalize(term: str) -> str:
    if term.endswith('.') or term.endswith(','):
        term = term[:-1]

    if term.startswith('('):
        term = term[1:]

    if term.endswith(')'):
        term = term[:-1]

    return (term.replace(' - ', '-')
            .replace(' %', '%')
            .replace(' . ', '.')
            .replace(' , ', ', ')
            .replace(' ’s', '’s')
            .replace('" ', '"')
            .replace(' "', '"')
            .replace('« ', '«')
            .replace(' »', '»')
            .replace('“ ', '“')
            .replace(" 's", "'s")
            .replace(' ”', '”')
            .replace('.(', '. (')
            .replace('( ', '(')
            .replace(' / ', '/')
            .replace(' )', ')')).strip()


class RobertaTermExtractor(TermExtractor):
    model = AutoModelForTokenClassification.from_pretrained(
        'aiwannafly/semantics-analysis-term-extractor', num_labels=len(LABEL_LIST))
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruRoberta-large', add_prefix_space=True)

    def __init__(self, device: str):
        self.model.to(device)

    def process(self, text: str) -> List[str]:
        words = text.split()

        tokenized_input = self.tokenizer(words, is_split_into_words=True, return_tensors='pt')

        with inference_mode():
            model_output = self.model(**tokenized_input)

        predicted = argmax(model_output['logits'][0], dim=1)[1:-1]
        word_ids = tokenized_input.word_ids()[1:-1]

        curr_idx = 0
        predicted_tags = []
        for i in range(len(words)):
            counts = {k: 0 for k in LABEL_LIST}
            while curr_idx < len(predicted) and word_ids[curr_idx] == i:
                counts[int2tag(predicted[curr_idx])] += 1
                curr_idx += 1
            # predicted_tag = list(sorted(counts.items(), key=lambda item: item[1], reverse=True))[0][0]

            if counts['I-TERM'] > 0:
                predicted_tag = 'I-TERM'
            elif counts['B-TERM'] > 0:
                predicted_tag = 'B-TERM'
            else:
                predicted_tag = 'O'
            predicted_tags.append(predicted_tag)

        assert len(predicted_tags) == len(words)

        terms = []

        curr_term_words = []

        for i in range(len(predicted_tags)):
            if predicted_tags[i] == 'I-TERM':
                if i == 0 or predicted_tags[i - 1] == 'O':
                    predicted_tags[i] = 'B-TERM'

        for i, tag in enumerate(predicted_tags):
            if tag == 'B-TERM':
                if curr_term_words:
                    terms.append(normalize(' '.join(curr_term_words)))
                curr_term_words = [words[i]]
            elif tag == 'O':
                if curr_term_words:
                    terms.append(normalize(' '.join(curr_term_words)))
                    curr_term_words = []
            else:
                curr_term_words.append(words[i])

        if curr_term_words:
            terms.append(normalize(' '.join(curr_term_words)))

        return list(set(terms))


def main():
    extractor = RobertaTermExtractor('cpu')

    terms = extractor.process('Всё-таки обработка предложений сильно завязана на предшествующий морфологический анализ.')

    print(terms)


if __name__ == '__main__':
    main()
