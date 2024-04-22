from typing import List, Optional

from rich.progress import Progress

from semantics_analysis.entities import ClassifiedTerm
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.ontology_utils import loaded_classes, term_metadata_by_class
from semantics_analysis.term_extraction.classified_term_extractor import ClassifiedTermExtractor


NOT_FOUND_RESPONSE = '—'


class LLMClassifiedTermExtractor(ClassifiedTermExtractor):
    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)
        self.prompt_by_class = {}

        with open('prompts/term_classification.txt', 'r', encoding='utf-8') as f:
            self.classification_prompt_template = f.read().strip()

        with open('prompts/term_extraction.txt', 'r', encoding='utf-8') as f:
            self.base_extraction_prompt_template = f.read().strip()

        for term_class in loaded_classes:
            self.prompt_by_class[term_class] = self._build_prompt(term_class)

    def __call__(self, text: str, progress: Progress) -> List[ClassifiedTerm]:

        total = len(loaded_classes)

        extraction = progress.add_task(description=f'Extracting terms by class 0/{total}', total=total)

        classes_by_term = dict()

        for idx, term_class in enumerate(loaded_classes):
            prompt = self.prompt_by_class[term_class]

            prompt = prompt.replace('{input}', text)

            try:
                response = self.llm_agent(prompt, max_new_tokens=100, stop_sequences=['.', '\n', NOT_FOUND_RESPONSE])
            except Exception as e:
                progress.remove_task(extraction)
                raise e

            if term_class == 'Task':
                print(prompt)
                print(response)

            progress.update(extraction, description=f'Extracting terms by class {idx + 1}/{total}', advance=1)

            if response == NOT_FOUND_RESPONSE:
                continue

            if response.endswith('.'):
                response = response[:-1]

            found_terms = response.split(', ')

            if not found_terms:
                continue

            for term in found_terms:
                if term in classes_by_term:
                    classes_by_term[term].append(term_class)
                else:
                    classes_by_term[term] = [term_class]

        progress.remove_task(extraction)

        ambiguous_terms = [t for t, classes in classes_by_term.items() if len(classes) > 1]

        if ambiguous_terms:
            total = len(ambiguous_terms)
            resolving = progress.add_task(description=f'Resolving ambiguous terms 0/{total}', total=total)

            for idx, term in enumerate(ambiguous_terms):
                classes = classes_by_term[term]

                print(f'{term} -> {classes}')

                try:
                    term_class = self._resolve_class(text, term, classes)
                except Exception as e:
                    progress.remove_task(resolving)
                    raise e

                classes_by_term[term] = [term_class]

                progress.update(resolving, description=f'Resolving ambiguous terms {idx + 1}/{total}')

            progress.remove_task(resolving)

        return [ClassifiedTerm(classes[0], t, len(text), text) for t, classes in classes_by_term.items()]

    def _resolve_class(self, text: str, term: str, classes: List[str]) -> str:
        if len(classes) == 1:
            return classes[0]

        class_descriptions = ''
        for class_ in classes:
            class_name = term_metadata_by_class[class_]['name']

            class_descriptions += f'- {class_}: {class_name}\n'

        class_descriptions = class_descriptions.strip()

        prompt = self.classification_prompt_template
        prompt = prompt.replace('{context}', text)
        prompt = prompt.replace('{term}', term)
        prompt = prompt.replace('{class_descriptions}', class_descriptions)
        prompt = prompt.replace('{classes}', ', '.join(classes))

        response = self.llm_agent(prompt, max_new_tokens=5, stop_sequences=['.', '\n', ','])

        # if term == 'GNMT':
        #     print(prompt)
        #     print(response)

        for class_ in classes:
            if class_ in response:
                return class_

        # failed to get correct response from LLM
        return classes[0]

    def _build_prompt(self, term_class: str) -> str:
        metadata = term_metadata_by_class[term_class]
        class_name = metadata['name']

        examples = '```\n'

        for example in metadata['examples']:
            text = example['text']
            terms = ', '.join(example['terms'])

            if not terms:
                terms = NOT_FOUND_RESPONSE
            else:
                terms += '.'

            examples += (f'Текст: "{text}"\n'
                         f'Термины класса "{class_name}": {terms}\n'
                         f'```\n')

        examples = examples.strip()

        prompt_template = self.base_extraction_prompt_template
        prompt_template = prompt_template.replace('{class}', class_name)
        prompt_template = prompt_template.replace('{examples}', examples)

        return prompt_template


def main():
    extractor = LLMClassifiedTermExtractor()

    text = 'GNMT есть система машинного перевода (NMT) компании Google, которая использует нейросеть (ANN) для повышения точности и скорости перевода, и в частности для создания лучших, более естественных вариантов перевода текста в Google Translate.'

    with Progress() as progress:
        found_terms = extractor(text, progress)

    for term in found_terms:
        print(term)


if __name__ == '__main__':
    main()
