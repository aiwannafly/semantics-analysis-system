from typing import List

from colorama import Fore, Style
from colorama import init as colorama_init

from semantics_analysis.config import load_config, Config
from semantics_analysis.entities import Relation, ClassifiedTerm
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor
from semantics_analysis.term_classification.roberta_term_classifier import RobertaTermClassifier
from semantics_analysis.term_classification.term_classifier import TermClassifier
from semantics_analysis.term_extraction.roberta_term_extractor import RobertaTermExtractor
from semantics_analysis.term_extraction.term_extractor import TermExtractor
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ComputerScienceTermPostProcessor
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTermPostProcessor
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from spinner import Spinner
import inquirer
import nltk

LOG_STYLE = Style.DIM
TERM_STYLE = Fore.LIGHTCYAN_EX
LABELED_TERM_STYLE = Fore.CYAN
PREDICATE_STYLE = Style.BRIGHT
SEPARATOR_STYLE = Style.DIM


def render_term(term: ClassifiedTerm) -> str:
    return f'{LABELED_TERM_STYLE}({term.value}: {term.class_}){Style.RESET_ALL}'


def render_relation(relation: Relation) -> str:
    # some specific code to render relation in terminal

    header_len = len('[      INPUT     ]:')

    term1, term2 = relation.term1, relation.term2

    term1_len, term2_len = len(f'({term1.value}: {term1.class_})'), len(f'({term2.value}: {term2.class_})')

    space_len = 10

    res = ''

    edge_len = term1_len - int(term1_len / 2) - 1 + space_len + int(term2_len / 2)

    edge_start_len = int((edge_len - len(relation.predicate)) / 2)

    edge_end_len = edge_len - edge_start_len - len(relation.predicate)

    edge = '—' * edge_start_len + relation.predicate + '—' * edge_end_len

    res += ' ' * (int(term1_len / 2)) + '┌' + edge + '┐' + '\n'

    res += ' ' * (header_len + int(term1_len / 2)) + '|' + ' ' * len(edge) + '|' + '\n'

    res += ' ' * header_len + render_term(term1) + ' ' * space_len + render_term(term2)

    return res


def analyze_text(
        text: str,
        split_on_sentences: bool,
        term_extractor: TermExtractor,
        term_classifier: TermClassifier,
        relation_extractor: RelationExtractor,
        term_postprocessors: List[TermPostProcessor]
):
    text = text.replace('�', '').strip()

    print()

    with Spinner():
        terms = term_extractor(text)

    if not terms:
        print(f'{LOG_STYLE}[ TERMS NOT FOUND]\n')
        return

    offset = 0
    labeled_text = text

    for term in terms:
        prev_len = len(labeled_text)
        labeled_text = (labeled_text[:term.start_pos + offset] + f'{TERM_STYLE}{term.value}{Style.RESET_ALL}'
                        + labeled_text[term.end_pos + offset:])
        offset += len(labeled_text) - prev_len

    print(f'{LOG_STYLE}[   FOUND TERMS  ]{Style.RESET_ALL}: {labeled_text}\n')

    with Spinner():
        labeled_terms = term_classifier(text, terms)

        for term_postprocessor in term_postprocessors:
            labeled_terms = term_postprocessor(labeled_terms)

    offset = 0
    labeled_text = text

    for term in labeled_terms:
        prev_len = len(labeled_text)

        labeled_text = (labeled_text[:term.start_pos + offset] + render_term(term) + labeled_text[term.end_pos + offset:])

        offset += len(labeled_text) - prev_len

    print(f'{LOG_STYLE}[CLASSIFIED TERMS]{Style.RESET_ALL}: {labeled_text}\n')

    rel_count = 0

    if split_on_sentences:
        text_and_terms = []
        text_parts = nltk.tokenize.sent_tokenize(text)
        
        for text_part in text_parts:
            start_pos = text.index(text_part)
            end_pos = start_pos + len(text_part)

            part_terms = [t for t in labeled_terms if t.start_pos >= start_pos and t.end_pos <= end_pos]

            text_and_terms.append((text_part, part_terms))
    else:
        text_and_terms = [(text, labeled_terms)]

    for text_part, labeled_terms in text_and_terms:
        relations = relation_extractor(text_part, labeled_terms)

        while True:
            with Spinner():
                relation = next(relations, None)

            if not relation:
                break

            rel_count += 1

            log_header = '[   RELATION {: 4}]'.format(rel_count)
            print(f'{LOG_STYLE}{log_header}{Style.RESET_ALL}:' + render_relation(relation))
            print()
            print()

    if rel_count == 0:
        print(f'{LOG_STYLE}[  NO RELATIONS  ]{Style.RESET_ALL}\n')


def main():
    colorama_init()

    app_config = load_config('config.yml')

    term_extractor = RobertaTermExtractor(app_config.device)

    term_postprocessors = [
        ComputerScienceTermPostProcessor(),
        MergeCloseTermPostProcessor()
    ]

    term_classifier = RobertaTermClassifier(app_config.device)

    relation_extractor = LLMRelationExtractor(
        prompt_template_path='prompts/relation_extraction.txt',
        huggingface_hub_token=app_config.huggingface_hub_token,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses
    )

    while True:
        text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')

        analyze_text(
            text,
            app_config.split_on_sentences,
            term_extractor,
            term_classifier,
            relation_extractor,
            term_postprocessors
        )

        question = inquirer.questions.List(
            name='answer',
            message='Продолжить анализировать тексты дальше?',
            choices=['Нет.', 'Да.']
        )

        answer = inquirer.prompt([question])['answer']

        if answer == 'Да.':
            continue
        else:
            break


if __name__ == '__main__':
    main()
