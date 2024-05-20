from typing import Optional

import spacy

from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer


NON_LEADING_POS = {'ADV', 'ADP', 'CCONJ', 'PUNCT'}
NON_TRAILING_POS = {'VERB', 'ADV', 'ADP', 'CCONJ', 'PUNCT'}


class LLMTermNormalizer(TermNormalizer):

    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        self.cached_results = {}

        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)

        with open('prompts/normalization.txt', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

        self.nlp = spacy.load("ru_core_news_sm")
        self.nlp.disable_pipes(["parser", "attribute_ruler", "lemmatizer"])

    def __call__(self, term: str) -> str:
        if not term:
            return term

        cached_result = self.cached_results.get(term, None)

        if cached_result is not None:
            return cached_result

        pos_tags = [token.pos_ for token in self.nlp(term)]

        if not pos_tags:
            return term

        while pos_tags and pos_tags[0] in NON_LEADING_POS:
            pos_tags.pop(0)
            term = term[term.find(' '):].strip()

        if not pos_tags:
            return term

        while pos_tags and pos_tags[-1] in NON_TRAILING_POS:
            pos_tags.pop()
            term = term[:term.rfind(' ')].strip()

        if not pos_tags:
            return term

        prompt = self.prompt_template.replace('{input}', term)

        normalized = self.llm_agent(
            prompt,
            stop_sequences=['\n', '(', '.'],
            max_new_tokens=50
        ).replace('(', '')

        while normalized.endswith('.'):
            normalized = normalized[:-1]

        self.cached_results[term] = normalized
        return normalized


def main():
    term = "техникой вероятностного латентно-семантического индекса"

    normalizer = LLMTermNormalizer()

    normalized = normalizer(term)

    print(normalized)

    a = [1, 2, 3]

    print(a[5:])


if __name__ == '__main__':
    main()
