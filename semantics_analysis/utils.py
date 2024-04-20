from typing import Dict, List, Tuple, Optional, Set

from colorama import Style, Fore
from rich.progress import Progress, TextColumn, BarColumn
from rich.table import Table
from rich.console import Console
from rich.text import Text

from semantics_analysis.entities import Term, GroupedTerm, ClassifiedTerm, Relation

LOG_STYLE = Style.DIM
TERM_STYLE = Fore.CYAN
LABELED_TERM_STYLE = Fore.CYAN
LABELED_DICT_TERM_STYLE = Fore.LIGHTMAGENTA_EX
GROUPED_TERM_STYLE = Fore.LIGHTGREEN_EX
PREDICATE_STYLE = Style.BRIGHT
SEPARATOR_STYLE = Style.DIM


def log(*messages: str):
    print(*messages)


def union_groups(
        considered_groups: List[GroupedTerm]
) -> List[GroupedTerm]:
    if len(considered_groups) < 2:
        return considered_groups

    final_groups: Set[GroupedTerm] = set()

    for i in range(len(considered_groups)):
        for j in range(i + 1, len(considered_groups)):
            group1, group2 = considered_groups[i], considered_groups[j]

            if group1.class_ != group2.class_:
                final_groups.add(group1)
                final_groups.add(group2)
                continue

            term_vals1 = set(t.value.lower() for t in group1.items)
            term_vals2 = set(t.value.lower() for t in group2.items)

            intersection = [t for t in term_vals1 if t in term_vals2]

            if intersection:
                final_groups.add(GroupedTerm(group1.class_, group1.items + group2.items, normalize=False))
            else:
                final_groups.add(group1)
                final_groups.add(group2)

    if len(final_groups) < len(considered_groups):
        return union_groups(list(final_groups))

    return list(final_groups)


def normalize_relations(groups: List[GroupedTerm], relations: List[Relation]) -> List[Relation]:
    new_terms: Dict[Relation, Tuple[Optional[ClassifiedTerm], Optional[ClassifiedTerm]]] = {}

    for rel in relations:
        new_terms[rel] = (None, None)

    for group in groups:
        if len(group.items) < 2:
            continue

        main_item = group.items[0]
        other_items = set(group.items[1:])

        other_items_rels = [r for r in relations if r.term1 in other_items or r.term2 in other_items]

        for rel in other_items_rels:
            new_term1, new_term2 = new_terms[rel]

            if rel.term1 in other_items:
                new_term1 = main_item

            if rel.term2 in other_items:
                new_term2 = main_item
            new_terms[rel] = new_term1, new_term2

    norm_relations = []

    for rel in relations:
        new_term1, new_term2 = new_terms[rel]

        term1 = new_term1 if new_term1 else rel.term1
        term2 = new_term2 if new_term2 else rel.term2

        norm_relations.append(Relation(term1, rel.predicate, term2))

    return norm_relations


def normalize_groups(groups: List[GroupedTerm]) -> List[GroupedTerm]:
    norm_groups = []

    for group in groups:
        if len(group.items) < 2:
            norm_groups.append(group)
            continue

        items = []
        values = set()

        for item in group.items:
            value = item.value.lower()

            if value not in values:
                values.add(value)

                items.append(item)

        norm_groups.append(GroupedTerm(group.class_, items, normalize=False))

    return norm_groups


def render_term(term: ClassifiedTerm, style: str = LABELED_TERM_STYLE) -> str:
    if style == LABELED_TERM_STYLE and term.source == 'dict':
        style = LABELED_DICT_TERM_STYLE

    return f'{style}({term.value}: {term.class_}){Style.RESET_ALL}'


def render_grouped_term(term: GroupedTerm) -> str:
    values = ', '.join([t.value for t in term.items])

    return f'{GROUPED_TERM_STYLE}({values}: {term.class_}){Style.RESET_ALL}'


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

    res += (' ' * header_len
            + render_term(term1, GROUPED_TERM_STYLE)
            + ' ' * space_len
            + render_term(term2, GROUPED_TERM_STYLE))

    return res


def log_grouped_terms(grouped_terms: List[GroupedTerm]):
    non_single_terms = [t for t in grouped_terms if t.size() > 1]

    term_count = 1
    for term in non_single_terms:
        log_header = '[   GROUP    {: 4}]'.format(term_count)
        term_count += 1

        log(f'{LOG_STYLE}{log_header}{Style.RESET_ALL}: ' + render_grouped_term(term))
        log()


def log_extracted_terms(text: str, terms: List[Term]):
    if not terms:
        log(f'{LOG_STYLE}[ TERMS NOT FOUND]\n')
        return

    offset = 0
    labeled_text = text

    for term in terms:
        prev_len = len(labeled_text)
        labeled_text = (labeled_text[:term.start_pos + offset] + f'{TERM_STYLE}{term.value}{Style.RESET_ALL}'
                        + labeled_text[term.end_pos + offset:])
        offset += len(labeled_text) - prev_len

    log(f'{LOG_STYLE}[   FOUND TERMS  ]{Style.RESET_ALL}: {labeled_text}\n')


def log_labeled_terms(text: str, labeled_terms: List[ClassifiedTerm]):
    offset = 0
    labeled_text = text

    for term in labeled_terms:
        prev_len = len(labeled_text)

        labeled_text = (
                    labeled_text[:term.start_pos + offset] + render_term(term) + labeled_text[term.end_pos + offset:])

        offset += len(labeled_text) - prev_len

    log(f'{LOG_STYLE}[CLASSIFIED TERMS]{Style.RESET_ALL}: {labeled_text}\n')


def log_found_relations(found_relations: List[Relation]):
    if not found_relations:
        log(f'{LOG_STYLE}[  NO RELATIONS  ]{Style.RESET_ALL}\n')
        return

    for idx, relation in enumerate(found_relations):
        log_header = '[   RELATION {: 4}]'.format(idx + 1)
        log(f'{LOG_STYLE}{log_header}{Style.RESET_ALL}:' + render_relation(relation))
        log()
        log()


def log_term_predictions(term_predictions: Optional[List[Tuple[str, str, int, int, int]]]):
    table = Table(title='Term predictions')
    table.add_column(header='Word')
    table.add_column(header='Predicted label')
    table.add_column(header='O')
    table.add_column(header='B-TERM')
    table.add_column(header='I-TERM')

    for word, label, p1, p2, p3 in term_predictions:
        style2, style3 = None, None

        max_p = max(p1, p2, p3)

        max_style = 'rgb(9,176,179)'

        if p2 == max_p:
            style2 = max_style
        elif p3 == max_p:
            style3 = max_style

        table.add_row(
            word,
            label,
            Text(str(p1)),
            Text(str(p2), style=style2),
            Text(str(p3), style=style3),
        )

    console = Console()
    console.print(table)


def log_class_predictions(predictions_by_term: Dict[Term, List[Tuple[str, float]]]):
    colors = ['green', 'magenta', 'purple3']

    for term, predictions in predictions_by_term.items():
        log()
        log(f'{Style.DIM}—————————————————————————————————————————————————————————————{Style.RESET_ALL}')
        log()
        log(f'[{term.start_pos}] {TERM_STYLE}{term.value}{Style.RESET_ALL}\n')

        predictions = sorted(predictions, key=lambda pair: pair[1], reverse=True)[:len(colors)]

        count = 0
        for class_, p in predictions:
            color = colors[count]
            count += 1

            p = int(p * 100) / 100.0

            diff = 20 - len(f'{class_}: {p}')

            description = f'[{color}]{class_}: {p}'

            if diff > 0:
                description = ' ' * diff + description

            with Progress(TextColumn(text_format=description, justify='right'),
                          BarColumn(complete_style=color)) as progress:
                class_task = progress.add_task(total=1.0, description='')

                progress.update(class_task, description='', advance=p)

        log()
