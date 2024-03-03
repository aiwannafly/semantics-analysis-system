from typing import List

from torch import inference_mode
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from semantics_analysis.entities import Term
from semantics_analysis.term_classification.term_classifier import TermClassifier


label_list = [
    'Method',
    'Activity',
    'Object',
    'Person',
    'Task',
    'Organization',
    'Model',
    'Metric',
    'Value',
    'Date',
    'Lang',
    'Dataset',
    'Application',
    'Science'
]

id2label = {i: label_list[i] for i in range(len(label_list))}

label2id = {label_list[i]: i for i in range(len(label_list))}


class RobertaTermClassifier(TermClassifier):
    tokenizer = AutoTokenizer.from_pretrained('ai-forever/ruRoberta-large')
    model = AutoModelForSequenceClassification.from_pretrained(
        'aiwannafly/semantics-analysis-term-classifier',
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )

    def __init__(self, device: str):
        self.model.to(device)

    def process(self, text: str, terms: List[str]) -> List[Term]:
        labeled_terms = []

        for term in terms:
            markup_text = text.replace(term, f'<term>{term}</term>')

            with inference_mode():
                outputs = self.model.forward(**self.tokenizer(markup_text, return_tensors='pt').to(device='cpu'))

            idx = outputs.logits.argmax(dim=1)[0].item()

            class_ = id2label[idx]

            labeled_terms.append(Term(class_, term))

        return labeled_terms


def main():
    text = 'Базис является максимальным по включению набором линейно-независимых векторов пространства.'
    terms = ['Базис', 'линейно-независимых векторов']

    classifier = RobertaTermClassifier('cpu')

    labeled_terms = classifier.process(text, terms)

    print(labeled_terms)


if __name__ == '__main__':
    main()
