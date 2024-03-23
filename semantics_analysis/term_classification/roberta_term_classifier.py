from typing import List, Dict, Tuple

import torch
from torch import inference_mode
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from semantics_analysis.entities import Term, ClassifiedTerm
from semantics_analysis.term_classification.term_classifier import TermClassifier

label_list = [
    'Method',
    'Activity',
    'Science',
    'Object',
    'Person',
    'InfoResource',
    'Task',
    'Organization',
    'Environment',
    'Model',
    'Metric',
    'Value',
    'Application',
    'Date',
    'Lang',
    'Dataset'
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
        self.device = device
        self.model.to(device)

    def __call__(self, text: str, terms: List[Term]) -> List[ClassifiedTerm]:
        return self.run_and_save_predictions(text, terms, {})

    def run_and_save_predictions(
            self,
            text: str,
            terms: List[Term],
            predictions: Dict[Term, List[Tuple[str, float]]]
    ) -> List[ClassifiedTerm]:
        classified_terms = []

        for term in terms:
            markup_text = text[:term.start_pos] + '<term>' + term.value + '</term>' + text[term.end_pos:]

            with inference_mode():
                outputs = self.model.forward(**self.tokenizer(markup_text, return_tensors='pt').to(device=self.device))

            probs = torch.squeeze(torch.softmax(outputs.logits, dim=1))

            predictions[term] = [(class_, p.item()) for class_, p in zip(label_list, probs)]

            idx = outputs.logits.argmax(dim=1)[0].item()

            class_ = id2label[idx]

            classified_terms.append(ClassifiedTerm.from_term(class_, term))

        return classified_terms
