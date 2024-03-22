from typing import List, Optional, Iterator

from huggingface_hub import InferenceClient

from semantics_analysis.entities import Relation, ClassifiedTerm
from semantics_analysis.relation_extractor.ontology_utils import predicates_by_class_pair, prompt_metadata_by_class_pair
from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor


class LLMRelationExtractor(RelationExtractor):

    def __init__(self,
                 prompt_template_path: str,
                 huggingface_hub_token: str,
                 model: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 show_explanation: bool = False,
                 log_prompts: bool = False,
                 log_llm_responses: bool = False):
        self.llm = InferenceClient(model=model, timeout=8, token=huggingface_hub_token)

        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read().strip()

        self.show_explanation = show_explanation
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses

    def __call__(self, text: str, terms: List[ClassifiedTerm]) -> Iterator[Relation]:
        # we seek only for binary relations
        # so every pair of terms can be checked

        # we use the knowledge from ontology and check only those pairs
        # that can be in a relation

        # so its normal to check for relation between Task and Method,
        # but it does not make sense to check for relation between InfoResource and Metric

        terms_count = len(terms)

        for i in range(terms_count):
            for j in range(i + 1, terms_count):
                term1, term2 = terms[i], terms[j]

                if term1.value == term2.value:
                    continue

                class1, class2 = term1.class_, term2.class_

                if (class1, class2) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class1, class2)]
                elif (class2, class1) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class2, class1)]

                    term1, term2 = term2, term1
                else:
                    predicates = []

                if not predicates:
                    continue

                predicate = self.detect_predicate(term1, term2, text)

                if not predicate:
                    continue

                yield Relation(term1, predicate, term2)

    def get_pairs_to_consider(self, terms: List[ClassifiedTerm]) -> List:
        terms_count = len(terms)

        pairs_to_consider = []

        for i in range(terms_count):
            for j in range(i + 1, terms_count):
                term1, term2 = terms[i], terms[j]

                if term1.value == term2.value:
                    continue

                class1, class2 = term1.class_, term2.class_

                if (class1, class2) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class1, class2)]
                elif (class2, class1) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class2, class1)]

                    term1, term2 = term2, term1
                else:
                    predicates = []

                if not predicates:
                    continue

                pairs_to_consider.append((term1, term2))

        return pairs_to_consider

    def analyze_term_pairs(
            self,
            text: str,
            term_pairs: List
    ) -> Iterator[Optional[Relation]]:
        for term1, term2 in term_pairs:
            predicate = self.detect_predicate(term1, term2, text)

            if predicate is None:
                yield None
            else:
                yield Relation(term1, predicate, term2)

    def detect_predicate(self, term1: ClassifiedTerm, term2: ClassifiedTerm, text: str) -> Optional[str]:

        prompt = self.create_llm_prompt(term1, term2, text)

        response = self.llm.text_generation(prompt, do_sample=False, max_new_tokens=100, stop_sequences=['.']).strip()

        if self.log_prompts:
            print(f'[INPUT PROMPT]: {prompt}\n')

        if self.log_llm_responses:
            print(f'[LLM RESPONSE]: {response}\n')

        if response.startswith('none'):
            return None

        predicates = predicates_by_class_pair[(term1.class_, term2.class_)]

        for predicate in predicates:
            if predicate in response and 'Нет' not in response and 'нет' not in response: #.startswith(predicate):
                return predicate

        return None

    def create_llm_prompt(self, term1: ClassifiedTerm, term2: ClassifiedTerm, text: str) -> str:
        class1, class2 = term1.class_, term2.class_

        prompt_metadata = prompt_metadata_by_class_pair[(class1, class2)]

        relations_list = ''
        examples_list = ''
        counter = 1

        predicates = []
        questions_list = ''

        for predicate, metadata in prompt_metadata.items():
            predicates.append(predicate)

            description = metadata['yes']['description']
            relations_list += f' - {predicate} : {description}\n'

            for answer in ['yes', 'no']:
                description = metadata[answer]['description']

                examples_list += f'{counter}. В этих примерах {description}:\n'

                for example in metadata[answer]['examples']:
                    example_text = example['text']

                    if class1 != class2:
                        example_term1 = example[class1]
                        example_term2 = example[class2]
                    else:
                        example_term1 = example[class1 + '_1']
                        example_term2 = example[class1 + '_2']

                    reply = 'нет.' if answer == 'no' else 'да'

                    examples_list += '```\n'
                    examples_list += (f'Текст: {example_text}\n'
                                      f'Термин {class1}: {example_term1}\n'
                                      f'Термин {class2}: {example_term2}\n'
                                      f'Есть ли подходящее отношение между терминами "{example_term1}" и "{example_term2}" в этом тексте? {reply}\n')

                    if answer == 'yes':
                        if not self.show_explanation:
                            examples_list += f'{predicate}.\n'
                        else:
                            examples_list += f'{predicate}\nОбъяснение: <объяснение>.\n'
                    examples_list += '```\n'

                    counter += 1

            questions_list += f' - Отношение {predicate}:\n'

            for question in prompt_metadata[predicate]['evaluation-questions']:
                questions_list += f'   - {question}\n'

        input_text = (f'Текст: {text}\n'
                      f'Термин {class1}: {term1.value}\n'
                      f'Термин {class2}: {term2.value}\n'
                      f'Есть ли подходящее отношение между терминами "{term1.value}" и "{term2.value}" в этом тексте?')

        prompt = self.prompt_template
        prompt = prompt.replace('{class1}', class1)
        prompt = prompt.replace('{class2}', class2)
        prompt = prompt.replace('{relations_list}', relations_list)
        prompt = prompt.replace('{examples_list}', examples_list)
        prompt = prompt.replace('{questions_list}', questions_list)
        prompt = prompt.replace('{relation_names}', str(predicates))
        prompt = prompt.replace('{input}', input_text)

        return prompt.strip()
