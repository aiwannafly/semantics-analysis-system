from typing import List, Optional, Iterator

from huggingface_hub import InferenceClient

from semantics_analysis.entities import Term, Relation
from semantics_analysis.relation_extractor.ontology_utils import predicates_by_class_pair, prompt_metadata_by_class_pair
from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor


PROMPT_TEMPLATE = '''
Твоя задача состоит в определении отношений между терминами.

Между терминами класса {class1} и {class2} возможны следующие отношения:
{relations_list}
Для ответа на вопрос о том, есть ли в указанном тексте отношение или нет, тебе помогут следующие вопросы:
{questions_list}
Примеры:
{examples_list}
Теперь скажи, какое отношение будет в этом случае.
Важно: давай ответ только в контексте указанного предложения. Не используй дополнительные знания, чтобы воссоздать связи, которых нет.
Если в рамках данного предложения связи между этими терминами нет, то и следует указать none.

Следуй формату из примеров. Тебе нужно указать ровно одно из возможных предложений.

{input}
'''.strip()


class LLMRelationExtractor(RelationExtractor):

    def __init__(self, huggingface_hub_token: str, log_prompts: bool = False, log_llm_responses: bool = False):
        self.llm = InferenceClient(model='mistralai/Mistral-7B-Instruct-v0.2', timeout=8, token=huggingface_hub_token)
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses

    def process(self, text: str, terms: List[Term]) -> Iterator[Relation]:
        # we seek only for binary relations
        # so every pair of terms can be checked

        # we use the knowledge from ontology and check only those pairs
        # that can be in a relation

        # so its normal to check for relation between Task and Method,
        # but it does not make sense to check for relation between InfoResource and Metric

        relations = []

        terms_count = len(terms)

        for i in range(terms_count):
            for j in range(i + 1, terms_count):
                term1, term2 = terms[i], terms[j]

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

    def detect_predicate(self, term1: Term, term2: Term, text: str) -> Optional[str]:

        prompt = self.create_llm_prompt(term1, term2, text)

        response = self.llm.text_generation(prompt, do_sample=False, max_new_tokens=10).strip()

        if self.log_prompts:
            print(f'[INPUT PROMPT]: {prompt}\n')

        if self.log_llm_responses:
            print(f'[LLM RESPONSE]: {response}\n')

        if response.startswith('none'):
            return None

        predicates = predicates_by_class_pair[(term1.class_, term2.class_)]

        for predicate in predicates:
            if response.startswith(predicate):
                return predicate

        return None

    @staticmethod
    def create_llm_prompt(term1: Term, term2: Term, text: str) -> str:
        class1, class2 = term1.class_, term2.class_

        prompt_metadata = prompt_metadata_by_class_pair[(class1, class2)]

        relations_list = ''
        examples_list = ''
        counter = 1

        for predicate, metadata in prompt_metadata['predicates'].items():
            description = metadata['description']
            try:
                example = metadata['examples'][0]
            except Exception:
                print(term1.class_, term2.class_, predicate)
                exit(-1)

            example_text = example['text']

            if class1 != class2:
                example_term1 = example[class1]
                example_term2 = example[class2]
            else:
                example_term1 = example[class1 + '_1']
                example_term2 = example[class1 + '_2']

            relations_list += f' - {predicate} : {description}\n'

            examples_list += f'{counter}. В этих примерах {description}:\n'
            examples_list += '```\n'
            examples_list += (f'Предложение: {example_text}\nТермин {class1}: {example_term1}\n'
                              f'Термин {class2}: {example_term2}\nОтношение: {predicate}\n')
            examples_list += '```\n'

            counter += 1

        questions_list = ''

        for question in prompt_metadata['evaluation-questions']:
            questions_list += f' - {question}\n'

        input_text = f'Предложение: {text}\nТермин {class1}: {term1.value}\nТермин {class2}: {term2.value}\nОтношение:'

        prompt = PROMPT_TEMPLATE
        prompt = prompt.replace('{class1}', class1)
        prompt = prompt.replace('{class2}', class2)
        prompt = prompt.replace('{relations_list}', relations_list)
        prompt = prompt.replace('{examples_list}', examples_list)
        prompt = prompt.replace('{questions_list}', questions_list)
        prompt = prompt.replace('{input}', input_text)

        return prompt.strip()


def main():
    extractor = LLMRelationExtractor('<place-token-here>')

    relations = extractor.process('Метод AdaGrad широко используется для задачи классификации.', [
        Term('Method', 'AdaGrad'), Term('Task', 'классификации')
    ])

    print(relations)

    relations = extractor.process('Метод Adam так или иначе включает в себя и другой метод: моменты Нестерова.', [
        Term('Method', 'Метод Adam'), Term('Method', value='моменты Нестерова')
    ])

    print(relations)

    relations = extractor.process('В рамках исследования мы пробовали два различных метода: классификатор на RoBERTa и генеративные нейронные сети', terms=[
        Term('Method', 'классификатор на RoBERTa'), Term('Method', value='генеративные нейронные сети')
    ])

    print(relations)


if __name__ == '__main__':
    main()
