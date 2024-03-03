from semantics_analysis.config import load_config
from semantics_analysis.entities import Relation, Term
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_classification.roberta_term_classifier import RobertaTermClassifier
from semantics_analysis.term_extraction.roberta_term_extractor import RobertaTermExtractor
from colorama import init as colorama_init
from colorama import Fore, Style

from spinner import Spinner


LOG_STYLE = Style.DIM
TERM_STYLE = Fore.LIGHTCYAN_EX
LABELED_TERM_STYLE = Fore.CYAN
PREDICATE_STYLE = Style.BRIGHT


def render_term(term: Term) -> str:
    return f'{LABELED_TERM_STYLE}({term.value}: {term.class_}){Style.RESET_ALL}'


def render_relation(relation: Relation) -> str:
    # some specific code to render relation in terminal

    term1, term2 = relation.term1, relation.term2

    term1_len, term2_len = len(f'({term1.value}: {term1.class_})'), len(f'({term2.value}: {term2.class_})')

    space = ' ' * 10

    res = render_term(term1) + space + render_term(term2) + '\n'

    header = '[      INPUT     ]:'

    for i in range(len(header) + int(term1_len / 2)):
        res += ' '
    res += '│'

    edge_len = term1_len - int(term1_len / 2) - 1 + len(space) + int(term2_len / 2)

    res += ' ' * edge_len

    res += '│\n'

    for i in range(len(header) + int(term1_len / 2)):
        res += ' '
    res += '└'

    edge_len = term1_len - int(term1_len / 2) - 1 + len(space) + int(term2_len / 2)

    edge_start_len = int((edge_len - len(relation.predicate)) / 2)

    edge_end_len = edge_len - edge_start_len - len(relation.predicate)

    edge = '—' * edge_start_len + relation.predicate + '—' * edge_end_len

    res += edge

    res += '┘'

    return res


def main():
    colorama_init()

    app_config = load_config('config.yml')

    term_extractor = RobertaTermExtractor(app_config.device)

    term_classifier = RobertaTermClassifier(app_config.device)

    relation_extractor = LLMRelationExtractor(
        app_config.huggingface_hub_token,
        app_config.log_prompts,
        app_config.log_llm_responses
    )

    text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')

    print()

    with Spinner():
        terms = term_extractor.process(text)

    if not terms:
        print(f'{LOG_STYLE}[ TERMS NOT FOUND]\n')
        return

    labeled_text = text
    for term in terms:
        labeled_text = labeled_text.replace(term, f'{TERM_STYLE}{term}{Style.RESET_ALL}')

    print(f'{LOG_STYLE}[   FOUND TERMS  ]{Style.RESET_ALL}: {labeled_text}\n')

    with Spinner():
        labeled_terms = term_classifier.process(text, terms)

    labeled_text = text
    for term in labeled_terms:
        labeled_text = labeled_text.replace(term.value, render_term(term))

    print(f'{LOG_STYLE}[CLASSIFIED TERMS]{Style.RESET_ALL}: {labeled_text}\n')

    relations = relation_extractor.process(text, labeled_terms)

    rel_count = 0
    while True:
        with Spinner():
            relation = next(relations, None)
            rel_count += 1

        if not relation:
            break

        print(f'{LOG_STYLE}[    RELATION {rel_count}  ]{Style.RESET_ALL}: ' + render_relation(relation))
        print()
        print()

    if rel_count == 0:
        print(f'{LOG_STYLE}[  NO RELATIONS  ]{Style.RESET_ALL}\n')


if __name__ == '__main__':
    main()
