from typing import List

from huggingface_hub import InferenceClient

from semantics_analysis.entities import ClassifiedTerm, GroupedTerm
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver


class LLMReferenceResolver(ReferenceResolver):

    def __init__(self,
                 huggingface_hub_token: str,
                 model: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 show_explanation: bool = False,
                 log_prompts: bool = False,
                 log_llm_responses: bool = False):
        with open('prompts/synonyms.txt', 'r', encoding='utf-8') as f:
            self.check_synonyms_prompt_template = f.read().strip()

        self.llm = InferenceClient(model=model, timeout=8, token=huggingface_hub_token)
        self.show_explanation = show_explanation
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses

    def __call__(self, terms: List[ClassifiedTerm], text: str) -> List[GroupedTerm]:
        terms_by_class = {term.class_ : [] for term in terms}

        for term in terms:
            terms_by_class[term.class_].append(term)

        grouped_terms = []
        for class_, terms in terms_by_class.items():
            group_by_term = {}

            curr_group_id = 0

            for i in range(len(terms)):
                for j in range(i + 1, len(terms)):
                    term1, term2 = terms[i], terms[j]

                    if term1.value.lower() == term2.value.lower() or self.are_synonyms(term1, term2, text):
                        if term1 in group_by_term:
                            group_by_term[term2] = group_by_term[term1]
                        elif term2 in group_by_term:
                            group_by_term[term1] = group_by_term[term2]
                        else:
                            group_by_term[term1] = curr_group_id
                            group_by_term[term2] = curr_group_id
                            curr_group_id += 1

            # add single terms
            grouped_terms.extend([GroupedTerm(class_, [t]) for t in terms if t not in group_by_term])

            terms_by_group = {}

            for term, group in group_by_term.items():
                if group not in terms_by_group:
                    terms_by_group[group] = [term]
                else:
                    terms_by_group[group].append(term)

            grouped_terms.extend([GroupedTerm(class_, terms) for terms in terms_by_group.values()])
        return grouped_terms

    def are_synonyms(self, term1: ClassifiedTerm, term2: ClassifiedTerm, text: str) -> bool:
        if term1.class_ != term2.class_:
            return False

        input_text = (f'Текст: {text}\n'
                      f'Являются ли "{term1.value}" и "{term2.value}" названиями одной и той же сущности в этом тексте?\n'
                      f'Ответ:')

        prompt = self.check_synonyms_prompt_template
        prompt = prompt.replace('{input}', input_text)

        if self.log_prompts:
            print(f'[INPUT PROMPT]: {prompt}\n')

        response = self.llm.text_generation(prompt, do_sample=False, max_new_tokens=2, stop_sequences=['.']).strip()

        if self.log_llm_responses:
            print(f'[ SYNONYMS ]: {response}\n')

        response = response.lower()

        return 'да' in response
