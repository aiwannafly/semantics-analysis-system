from colorama import Fore, Style
from colorama import init as colorama_init

from semantics_analysis.config import load_config
from semantics_analysis.entities import Relation, ClassifiedTerm
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_classification.roberta_term_classifier import RobertaTermClassifier
from semantics_analysis.term_extraction.roberta_term_extractor import RobertaTermExtractor
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ComputerScienceTermPostProcessor
from spinner import Spinner

LOG_STYLE = Style.DIM
TERM_STYLE = Fore.LIGHTCYAN_EX
LABELED_TERM_STYLE = Fore.CYAN
PREDICATE_STYLE = Style.BRIGHT


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


def main():
    colorama_init()

    app_config = load_config('config.yml')

    term_extractor = RobertaTermExtractor(app_config.device)

    term_postprocessor = ComputerScienceTermPostProcessor()

    term_classifier = RobertaTermClassifier(app_config.device)

    relation_extractor = LLMRelationExtractor(
        prompt_template_path='prompts/relation_extraction.txt',
        huggingface_hub_token=app_config.huggingface_hub_token,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses
    )

    text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')

    text = text.replace('�', '').strip()
    print()

    with Spinner():
        terms = term_extractor(text)

    if not terms:
        print(f'{LOG_STYLE}[ TERMS NOT FOUND]\n')
        return

    terms = sorted(terms, key=lambda t: t.start_pos)
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
        labeled_terms = term_postprocessor(labeled_terms)

    offset = 0
    labeled_terms = sorted(labeled_terms, key=lambda t: t.start_pos)
    labeled_text = text

    for term in labeled_terms:
        prev_len = len(labeled_text)

        labeled_text = (labeled_text[:term.start_pos + offset] + render_term(term) + labeled_text[term.end_pos + offset:])

        offset += len(labeled_text) - prev_len

    print(f'{LOG_STYLE}[CLASSIFIED TERMS]{Style.RESET_ALL}: {labeled_text}\n')

    relations = relation_extractor(text, labeled_terms)

    relations_by_first_term = {}

    rel_count = 0
    while True:
        with Spinner():
            relation = next(relations, None)
            rel_count += 1

        if not relation:
            break

        if relation.term1 not in relations_by_first_term:
            relations_by_first_term[relation.term1] = [relation]
        else:
            relations_by_first_term[relation.term1].append(relation)

        log_header = '[   RELATION {: 4}]'.format(rel_count)
        print(f'{LOG_STYLE}{log_header}{Style.RESET_ALL}:' + render_relation(relation))
        print()
        print()

    if rel_count == 0:
        print(f'{LOG_STYLE}[  NO RELATIONS  ]{Style.RESET_ALL}\n')

    # for first_term, relations in relations_by_first_term.items():
    #     if len(relations) > 2:
    #         print(first_term)


if __name__ == '__main__':
    main()
