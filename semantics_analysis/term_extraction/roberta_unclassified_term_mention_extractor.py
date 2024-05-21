from typing import List

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from semantics_analysis.entities import TermMention
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor, PredictionsType

NON_LEADING_POS = {'ADV', 'ADP', 'CCONJ', 'PUNCT', 'PRON'}
NON_TRAILING_POS = NON_LEADING_POS.union('VERB')

LABEL_LIST = ['O', 'B-TERM', 'I-TERM']
PUNCTUATION_SYMBOLS = {'.', ',', ':', '&', '?', '!', '—', '-', ')', '(', '[', ']', '{', '}', '"', "'", '«', '»'}
PREPOSITIONS = {'в', 'без', 'до', 'для', 'за', 'через', 'над', 'по', 'из', 'у', 'около', 'под', 'о', 'про', 'на', 'к',
                'перед', 'при', 'с', 'между'}
CONJUNCTIONS = {'и', 'или', 'а', 'но', 'либо', 'ни'}


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


class RobertaUnclassifiedTermExtractor(TermMentionExtractor):
    def __init__(self, device: str = 'cpu', min_term_prob: float = 0.1):
        self.model_path = 'aiwannafly/semantics-analysis-term-extractor'

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=len(LABEL_LIST)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ai-forever/ruRoberta-large',
            add_prefix_space=True
        )
        self.min_term_prob = min_term_prob * 100
        self.model.to(device)
        self.nlp = spacy.load("ru_core_news_sm")
        self.nlp.disable_pipes(["parser", "attribute_ruler", "lemmatizer"])

    def __call__(self, text: str, predictions: PredictionsType = None) -> List[TermMention]:
        doc = self.nlp(text)

        words = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]

        assert len(pos_tags) == len(words)

        tokenized_input = self.tokenizer(words, is_split_into_words=True, return_tensors='pt')

        with torch.inference_mode():
            model_output = self.model(**tokenized_input)

        probs_tensor = model_output['logits'][0]

        word_ids = tokenized_input.word_ids()[1:-1]
        probs = torch.squeeze(torch.softmax(probs_tensor, dim=1))

        probs = [(p1.item(), p2.item(), p3.item()) for (p1, p2, p3) in probs]
        probs = [(int(p1 * 100), int(p2 * 100), int(p3 * 100)) for (p1, p2, p3) in probs]

        curr_idx = 0
        labels = []
        for i in range(len(words)):
            word = words[i]

            word_predictions = []
            while curr_idx < len(word_ids) and word_ids[curr_idx] == i:
                word_predictions.append(probs[curr_idx])
                curr_idx += 1

            p_other, p_bterm, p_iterm = [max(probs[i] for probs in word_predictions) for i in range(3)]

            if word in PUNCTUATION_SYMBOLS - {'-'} or (p_bterm < self.min_term_prob and p_iterm < self.min_term_prob):
                predicted_tag = 'O'
            elif p_bterm > p_iterm:
                predicted_tag = 'B-TERM'
            else:
                predicted_tag = 'I-TERM'

            labels.append(predicted_tag)

            if predictions is not None:
                for p1, p2, p3 in word_predictions:
                    predictions.append((word, predicted_tag, p1, p2, p3))

        assert len(labels) == len(words)

        # turn leading I-TERMs into B-TERMs
        for i in range(len(labels)):
            if labels[i] == 'I-TERM':
                if i == 0 or labels[i - 1] == 'O':
                    labels[i] = 'B-TERM'

        for i in range(len(labels)):
            word, label, pos_tag = words[i], labels[i], pos_tags[i]

            is_last = i == len(labels) - 1

            if label == 'B-TERM' and pos_tag in NON_LEADING_POS:
                labels[i] = 'O'

                if not is_last and labels[i + 1] == 'I-TERM':
                    labels[i + 1] = 'B-TERM'

            is_term_end = 'TERM' in label and (is_last or labels[i + 1] == 'O')

            if is_term_end and pos_tag in NON_TRAILING_POS:
                labels[i] = 'O'

        terms = []

        curr_text = ''
        curr_term = ''
        is_under_term = False

        ontology_class = 'unknown'

        for label, token in zip(labels, words):
            remain_text = text[len(curr_text):]

            if label == 'B-TERM':
                is_under_term = True

                if curr_term:
                    terms.append(
                        TermMention(
                            value=curr_term.strip(),
                            text=text,
                            end_pos=len(curr_text),
                            ontology_class=ontology_class)
                    )
                    curr_term = ''
            elif label == 'O':
                is_under_term = False
                if curr_term:
                    terms.append(
                        TermMention(
                            value=curr_term.strip(),
                            text=text,
                            end_pos=len(curr_text),
                            ontology_class=ontology_class)
                    )
                    curr_term = ''

            for i in range(len(remain_text)):
                curr_text += remain_text[i]

                if is_under_term:
                    curr_term += remain_text[i]

                if curr_text.endswith(token):
                    break

        if curr_term:
            terms.append(
                TermMention(
                    value=curr_term.strip(),
                    text=text,
                    end_pos=len(curr_text),
                    ontology_class=ontology_class)
            )

        for term in terms:
            for s in PUNCTUATION_SYMBOLS:
                while term.value.endswith(s):
                    term.value = term.value[:-1]
                    term.end_pos -= 1
                
                while term.value.startswith(s):
                    term.value = term.value[1:]
                    term.start_pos += 1

        return terms


def main():
    term_extractor = RobertaUnclassifiedTermExtractor('cpu')

    terms = term_extractor('Всё-таки обработка предложений сильно завязана на предшествующий морфологический анализ.')

    print(terms)


if __name__ == '__main__':
    main()
