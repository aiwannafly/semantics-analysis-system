from typing import List, Optional, Tuple

from torch import argmax, inference_mode, softmax, squeeze
from transformers import AutoTokenizer, AutoModelForTokenClassification

from semantics_analysis.entities import Term
from semantics_analysis.term_extraction.term_extractor import TermExtractor

LABEL_LIST = ['O', 'B-TERM', 'I-TERM']
PUNCTUATION_SYMBOLS = [',', ':', '&', '?', '!', '-', '—', ')', '(', '[', ']', '{', '}', '"', "'", '«', '»']


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


class RobertaTermExtractor(TermExtractor):
    model = AutoModelForTokenClassification.from_pretrained(
        'aiwannafly/semantics-analysis-term-extractor', num_labels=len(LABEL_LIST))
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruRoberta-large', add_prefix_space=True)

    def __init__(self, device: str):
        self.model.to(device)

    def __call__(self, text: str,
                 predictions: Optional[List[Tuple[str, str, int, int, int]]] = None
                 ) -> List[Term]:
        preprocessed_text = text

        for s in PUNCTUATION_SYMBOLS:
            preprocessed_text = preprocessed_text.replace(s, ' ' + s + ' ')

        preprocessed_text = preprocessed_text.replace('. ', ' . ')

        if preprocessed_text.endswith('.'):
            preprocessed_text = preprocessed_text[:-1] + ' .'

        preprocessed_text = preprocessed_text.replace('  ', ' ')

        words = preprocessed_text.split()

        tokenized_input = self.tokenizer(words, is_split_into_words=True, return_tensors='pt')

        with inference_mode():
            model_output = self.model(**tokenized_input)

        predicted = argmax(model_output['logits'][0], dim=1)[1:-1]
        word_ids = tokenized_input.word_ids()[1:-1]

        probs = squeeze(softmax(model_output['logits'][0], dim=1))

        probs = [(p1.item(), p2.item(), p3.item()) for (p1, p2, p3) in probs]
        probs = [(int(p1 * 100), int(p2 * 100), int(p3 * 100)) for (p1, p2, p3) in probs]

        curr_idx = 0
        labels = []
        for i in range(len(words)):
            counts = {k: 0 for k in LABEL_LIST}
            word_predictions = []
            while curr_idx < len(predicted) and word_ids[curr_idx] == i:
                counts[int2tag(predicted[curr_idx].item())] += 1
                curr_idx += 1

                if predictions is not None:
                    word_predictions.append(probs[curr_idx])

            if counts['I-TERM'] > 0:
                predicted_tag = 'I-TERM'
            elif counts['B-TERM'] > 0:
                predicted_tag = 'B-TERM'
            else:
                predicted_tag = 'O'
            labels.append(predicted_tag)

            if predictions is not None:
                word = words[i]
                for p1, p2, p3 in word_predictions:
                    predictions.append((word, predicted_tag, p1, p2, p3))

        assert len(labels) == len(words)

        # turn leading I-TERMs into B-TERMs
        for i in range(len(labels)):
            if labels[i] == 'I-TERM':
                if i == 0 or labels[i - 1] == 'O':
                    labels[i] = 'B-TERM'

        terms = []

        curr_text = ''
        curr_term = ''
        is_under_term = False

        for label, token in zip(labels, words):
            remain_text = text[len(curr_text):]

            if label == 'B-TERM':
                is_under_term = True

                if curr_term:
                    terms.append(Term(value=curr_term.strip(), text=text, end_pos=len(curr_text)))
                    curr_term = ''
                start_pos = len(curr_text)
            elif label == 'O':
                is_under_term = False
                if curr_term:
                    terms.append(Term(value=curr_term.strip(), text=text, end_pos=len(curr_text)))
                    curr_term = ''

            for i in range(len(remain_text)):
                curr_text += remain_text[i]

                if is_under_term:
                    curr_term += remain_text[i]

                if curr_text.endswith(token):
                    break

        if curr_term:
            terms.append(Term(value=curr_term.strip(), text=text, end_pos=len(curr_text)))

        for term in terms:
            for s in PUNCTUATION_SYMBOLS:
                while term.value.endswith(s):
                    term.value = term.value[:-1]
                    term.end_pos -= 1

        return terms


def main():
    term_extractor = RobertaTermExtractor('cpu')

    terms = term_extractor('Всё-таки обработка предложений сильно завязана на предшествующий морфологический анализ.')

    print(terms)


if __name__ == '__main__':
    main()
