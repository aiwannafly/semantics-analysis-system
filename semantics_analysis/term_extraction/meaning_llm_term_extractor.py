import json
from typing import List, Optional, Dict, Tuple

import nltk
from rich.progress import Progress
from tqdm import tqdm

from semantics_analysis.entities import ClassifiedTerm
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.meaning_model.meaning import Meaning
from semantics_analysis.meaning_model.mpnet_meaning_model import MpnetMeaningModel
from semantics_analysis.ontology_utils import term_metadata_by_class
from semantics_analysis.term_extraction.classified_term_extractor import ClassifiedTermExtractor
from semantics_analysis.term_extraction.phrase_extractor import PhraseExtractor

MIN_SIMILARITY = 0.3
EXAMPLES_PER_CLASS = 2
MAX_CLASSES_IN_PROMPT = 6


class MeaningLLMTermExtractor(ClassifiedTermExtractor):

    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)

        self.phrase_extractor = PhraseExtractor()

        self.meaning_model = MpnetMeaningModel()

        with open('prompts/define_term_class.txt', 'r', encoding='utf-8') as f:
            self.classification_prompt_template = f.read().strip()

        with open('prompts/verify_term.txt', 'r', encoding='utf-8') as f:
            self.verification_prompt_template = f.read().strip()

        with open('metadata/terms_by_class.json', 'r', encoding='utf-8') as f:
            terms_by_class = json.load(f)

        self.class_by_term = {}

        try:
            with open('metadata/meaning_by_term.json', 'r', encoding='utf-8') as f:
                vector_by_term = json.load(f)
        except IOError as _:

            # file not found
            vector_by_term = {}
            for class_, terms in terms_by_class.items():

                for term in tqdm(terms):
                    vector_by_term[term] = self.meaning_model.get_meaning(term).values

            with open('metadata/meaning_by_term.json', 'w', encoding='utf-8') as wf:
                json.dump(vector_by_term, wf, indent=2, ensure_ascii=False)

        for class_, terms in terms_by_class.items():

            for term in terms:
                self.class_by_term[term] = class_

        self.meaning_by_term = {}
        for term, vector in vector_by_term.items():
            self.meaning_by_term[term] = Meaning(term, vector)

    def __call__(self, text: str, progress: Progress) -> List[ClassifiedTerm]:

        sentences = nltk.tokenize.sent_tokenize(text)

        if len(sentences) > 1:
            all_terms = []
            total = len(sentences)
            sentences_task = progress.add_task(description=f'Sentence 0/{total}', total=total)

            for idx, sent in enumerate(sentences):
                try:
                    all_terms.extend(self(sent, progress))
                except Exception as e:
                    progress.remove_task(sentences_task)
                    raise e

                progress.update(sentences_task, description=f'Sentence {idx + 1}/{total}', advance=1)

            progress.remove_task(sentences_task)

            return list(set(all_terms))

        phrases = self.phrase_extractor(text)

        # print(phrases)

        if not phrases:
            return []

        found_terms = []

        total = len(phrases)
        resolving = progress.add_task(description=f'Resolving terms 0/{total}', total=total)

        checked_phrases = set()

        for idx, phrase in enumerate(phrases):
            if phrase in checked_phrases:
                continue
            else:
                checked_phrases.add(phrase)

            examples_by_class = self._find_potential_classes(phrase)

            try:
                response = self._resolve_class(text, phrase, examples_by_class)
            except Exception as e:
                progress.remove_task(resolving)
                raise e

            progress.update(resolving, description=f'Resolving terms {idx + 1}/{total}', advance=1)

            if not response:
                continue

            class_, term = response

            try:
                verified = self._verify_term(text, term, class_)
            except Exception as e:
                progress.remove_task(resolving)
                raise e

            if verified:
                found_terms.append(ClassifiedTerm(class_, term, len(text), text))

        progress.remove_task(resolving)

        return list(set(found_terms))

    def _verify_term(self, text: str, term: str, class_: str) -> bool:
        desc = term_metadata_by_class[class_]['description']
        class_name = term_metadata_by_class[class_]['name']

        prompt = self.verification_prompt_template
        prompt = prompt.replace('{class}', class_name)
        prompt = prompt.replace('{description}', desc)
        prompt = prompt.replace('{text}', text)
        prompt = prompt.replace('{term}', term)

        response = self.llm_agent(
            prompt,
            max_new_tokens=1,
            stop_sequences=['.', '\n']
        )

        return 'да' in response.lower()

    def _resolve_class(self, text: str, term: str, examples_by_class: Dict[str, List[str]]) -> Optional[Tuple[str, str]]:
        class_descriptions = ''
        for class_, examples in examples_by_class.items():
            desc = term_metadata_by_class[class_]['description']
            name = term_metadata_by_class[class_]['name']

            examples = ', '.join(examples)

            class_descriptions += (f'- {class_}:\n'
                                   f'    Название: {name}\n'
                                   f'    Описание: {desc}\n'
                                   f'    Примеры: {examples}\n\n')

        class_descriptions = class_descriptions.strip()

        prompt = self.classification_prompt_template
        prompt = prompt.replace('{context}', text)
        prompt = prompt.replace('{term}', term)
        prompt = prompt.replace('{class_descriptions}', class_descriptions)
        prompt = prompt.replace('{classes}', ', '.join(examples_by_class))

        response = self.llm_agent(
            prompt,
            max_new_tokens=30,
            stop_sequences=['.', '(']
        ).replace('(', '').strip()

        # print(prompt)
        # print(term)
        # print(response)

        if response.startswith('нет'):
            return None

        while response.endswith('.'):
            response = response[:len(response) - 1]

        response = response.replace('да\n', '').strip()

        sep_idx = response.rfind(':')

        if sep_idx == -1:
            for class_ in examples_by_class:
                if class_ in response:
                    return class_, term
            return None

        term = response[sep_idx+1:].strip()

        class_part = response[:sep_idx].strip()

        for class_ in examples_by_class:
            if class_ in class_part:
                return class_, term

        # failed to get correct response from LLM
        return None

    def _find_potential_classes(self, phrase: str) -> Dict[str, List[str]]:
        phrase_meaning = self.meaning_model.get_meaning(phrase)

        candidates_classes = set()

        examples_by_class = {}

        similarity_by_term = {}

        for term, class_ in self.class_by_term.items():
            term_meaning = self.meaning_by_term.get(term, None)

            if not term_meaning:
                continue

            similarity = term_meaning.calculate_similarity(phrase_meaning)

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

        return {class_: sorted_examples_by_class[class_] for class_ in sorted_classes}


def main():
    extractor = MeaningLLMTermExtractor()

    print('Press q to stop.')

    while True:
        text = input("Enter text: ").strip()

        if text == 'q':
            break

        with Progress() as progress:
            found_terms = extractor(text, progress)

        for term in found_terms:
            print(f'{term.value} -> {term.class_}')

        print()


if __name__ == '__main__':
    main()
