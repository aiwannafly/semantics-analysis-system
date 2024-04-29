import json

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from semantics_analysis.meaning_model.meaning import Meaning
from semantics_analysis.meaning_model.meaning_model import MeaningModel

MIN_SIMILARITY = 0.3
EXAMPLES_PER_CLASS = 2
MAX_CLASSES_IN_PROMPT = 6


class MpnetMeaningModel(MeaningModel):
    model_id = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    def __init__(self):
        super().__init__()
        # model class should be obtained
        # this can be done via transformers library
        self.bert = AutoModel.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def get_meaning(self, text: str) -> Meaning:
        input_ids = self.tokenizer(text, return_tensors='pt')['input_ids']

        with torch.inference_mode():
            result = self.forward(input_ids).squeeze()

        values = [n.item() for n in result]

        return Meaning(text, values)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [N, 768]
        token_embeddings = self.bert(input_ids)[0]

        # [1, 768]
        token_embeddings_sums = torch.sum(token_embeddings, 1)

        return F.normalize(token_embeddings_sums, p=2, dim=1)


def main():
    model = MpnetMeaningModel()

    # meaning1 = model.get_meaning("солнце")
    # meaning2 = model.get_meaning("луна")
    #
    # print(meaning1.calculate_similarity(meaning2))

    with open('metadata/terms_by_class.json', 'r', encoding='utf-8') as f:
        terms_by_class = json.load(f)
    class_by_term = {}

    with open('metadata/meaning_by_term.json', 'r', encoding='utf-8') as f:
        vector_by_term = json.load(f)

    for class_, terms in terms_by_class.items():

        for term in terms:
            class_by_term[term] = class_

    meaning_by_term = {}
    for term, vector in vector_by_term.items():
        meaning_by_term[term] = Meaning(term, vector)

    print('Meanings are loaded.\n')

    print(f'Press q to quit.\n')

    while True:

        example = input('Enter phrase: ').strip()

        if example == 'q':
            break

        example_meaning = model.get_meaning(example)

        candidates_classes = set()

        examples_by_class = {}

        print('Searching...')

        similarity_by_term = {}

        for term, class_ in class_by_term.items():
            term_meaning = meaning_by_term[term]

            similarity = term_meaning.calculate_similarity(example_meaning)

            if similarity < MIN_SIMILARITY:
                continue

            similarity_by_term[term] = similarity

            candidates_classes.add(class_)

            if class_ not in examples_by_class:
                examples_by_class[class_] = [term]
            else:
                examples_by_class[class_].append(term)

        similarity_by_class = {}
        sorted_examples_by_class = {}
        for class_, examples in examples_by_class.items():
            sorted_examples_by_class[class_] = sorted(examples, key=lambda ex: similarity_by_term[ex], reverse=True)[:EXAMPLES_PER_CLASS]

            similarity_by_class[class_] = max(similarity_by_term[ex] for ex in sorted_examples_by_class[class_])

        sorted_classes = sorted(candidates_classes, key=lambda c: similarity_by_class[c], reverse=True)[:MAX_CLASSES_IN_PROMPT]

        print('Found the candidates:')

        for class_ in sorted_classes:
            examples = sorted_examples_by_class[class_]

            similarity = similarity_by_class[class_]

            similarity = int(similarity * 100)

            print(f'- {class_} ({similarity}%): {examples}')

        print()


if __name__ == '__main__':
    main()
