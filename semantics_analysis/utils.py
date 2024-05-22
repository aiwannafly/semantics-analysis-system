from typing import Dict, List, Tuple, Optional, Set, TypeVar, Callable, Union, Any

from colorama import Style, Fore
from rich.progress import Progress, TextColumn, BarColumn, ProgressColumn, GetTimeCallable, TaskID
from rich.table import Table
from rich.console import Console
from rich.text import Text

from semantics_analysis.entities import TermMention, Term, Relation, BoundedIterator
from semantics_analysis.ontology_utils import CLASS_ALIASES
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer

LOG_STYLE = Style.DIM
TERM_STYLE = Fore.CYAN
LABELED_TERM_STYLE = Fore.CYAN
LABELED_DICT_TERM_STYLE = Fore.LIGHTMAGENTA_EX
GROUPED_TERM_STYLE = Fore.LIGHTGREEN_EX
PREDICATE_STYLE = Style.BRIGHT
SEPARATOR_STYLE = Style.DIM
T = TypeVar('T')


def log(*messages: str):
    print(*messages)


def _align_description(desc: str):
    if len(desc) < 30:
        desc += ' ' * (30 - len(desc))

    return desc


class AlignedProgress(Progress):

    def __init__(self, *columns: Union[str, ProgressColumn], console: Optional[Console] = None,
                 auto_refresh: bool = True, refresh_per_second: float = 10, speed_estimate_period: float = 30.0,
                 transient: bool = False, redirect_stdout: bool = True, redirect_stderr: bool = True,
                 get_time: Optional[GetTimeCallable] = None, disable: bool = False, expand: bool = False) -> None:
        super().__init__(*columns, console=console, auto_refresh=auto_refresh, refresh_per_second=refresh_per_second,
                         speed_estimate_period=speed_estimate_period, transient=transient,
                         redirect_stdout=redirect_stdout, redirect_stderr=redirect_stderr, get_time=get_time,
                         disable=disable, expand=expand)

    def add_task(self, description: str, start: bool = True, total: Optional[float] = 100.0, completed: int = 0,
                 visible: bool = True, **fields: Any) -> TaskID:
        return super().add_task(_align_description(description), start, total, completed, visible, **fields)

    def update(self, task_id: TaskID, *, total: Optional[float] = None, completed: Optional[float] = None,
               advance: Optional[float] = None, description: Optional[str] = None, visible: Optional[bool] = None,
               refresh: bool = False, **fields: Any) -> None:
        super().update(task_id, total=total, completed=completed, advance=advance,
                       description=_align_description(description), visible=visible, refresh=refresh, **fields)

    def advance(self, task_id: TaskID, advance: float = 1) -> None:
        super().advance(task_id, advance)


def log_iterations(
        description: str,
        iterator: BoundedIterator[T],
        progress: Progress,
        item_handler: Callable[[T], None]
):
    step = 0

    task = progress.add_task(
        description=f'{description} {step}/{iterator.total}',
        total=iterator.total
    )

    try:
        for item in iterator.items:
            step += 1

            item_handler(item)

            progress.update(
                task,
                description=f'{description} {step}/{iterator.total}',
                advance=1
            )
    except Exception as e:
        progress.remove_task(task)
        raise e

    progress.remove_task(task)


def union_term_mentions(
        considered_terms: List[Term]
) -> List[Term]:
    if len(considered_terms) < 2:
        return considered_terms

    final_terms: Set[Term] = set()

    for i in range(len(considered_terms)):
        for j in range(i + 1, len(considered_terms)):
            term1, term2 = considered_terms[i], considered_terms[j]

            if term1.class_ != term2.class_:
                final_terms.add(term1)
                final_terms.add(term2)
                continue

            term_vals1 = set(t.norm_value.lower() for t in term1.mentions)
            term_vals2 = set(t.norm_value.lower() for t in term2.mentions)

            intersection = set(t for t in term_vals1 if t in term_vals2)

            # ignore common terms like 'модель', intersection should be specific
            intersection -= CLASS_ALIASES

            if intersection:
                all_mentions = term1.mentions + term2.mentions

                prev_mentions = set()

                mentions = []
                for mention in all_mentions:
                    lower = mention.norm_value.lower()

                    if lower in prev_mentions:
                        continue
                    prev_mentions.add(lower)
                    mentions.append(mention)

                term1.mentions = mentions
                term2.mentions = mentions
                term2.value = term1.value
                final_terms.add(term1)
            else:
                final_terms.add(term1)
                final_terms.add(term2)

    if len(final_terms) < len(considered_terms):
        return union_term_mentions(list(final_terms))

    return list(final_terms)


def normalize_relations(groups: List[Term], relations: List[Relation]) -> List[Relation]:
    new_terms: Dict[Relation, Tuple[Optional[Term], Optional[Term]]] = {}

    for rel in relations:
        new_terms[rel] = (None, None)

    for group in groups:
        if len(group.mentions) < 2:
            continue

        main_item = group.mentions[0]
        other_items = set(group.mentions[1:])

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


def normalize_groups(groups: List[Term]) -> List[Term]:
    norm_groups = []

    for term in groups:
        if len(term.mentions) < 2:
            norm_groups.append(term)
            continue

        items = []
        values = set()

        for item in term.mentions:
            value = item.value.lower()

            if value not in values:
                values.add(value)

                items.append(item)

        norm_groups.append(Term(term.class_, term.value, term.mentions))

    return norm_groups


def normalize_term_values(
        groups: List[Term],
        relations: List[Relation],
        normalizer: TermNormalizer,
        progress: Progress
) -> Tuple[List[Term], List[Relation]]:

    norm_groups = []

    total = len(groups)
    normalize_terms = progress.add_task(description=f'Normalizing terms 0/{total}', total=total)
    for idx, group in enumerate(groups):
        main = group.items[0]

        others = group.items[1:]

        norm_value = normalizer(main.value)

        progress.update(normalize_terms, description=f'Normalizing terms {idx+1}/{total}', advance=1)

        main = Term(main.class_, norm_value, main.end_pos, main.text)

        norm_groups.append(Term(main.class_, [main] + others))

    progress.remove_task(normalize_terms)

    norm_relations = []

    for rel in relations:
        norm_term1 = Term(rel.term1.class_, normalizer(rel.term1.value), rel.term1.end_pos, rel.term1.text)

        norm_term2 = Term(rel.term2.class_, normalizer(rel.term2.value), rel.term2.end_pos, rel.term2.text)

        norm_relations.append(Relation(norm_term1, rel.predicate, norm_term2))

    return norm_groups, norm_relations


def render_term(term: TermMention, style: str = LABELED_TERM_STYLE) -> str:
    if style == LABELED_TERM_STYLE and term.source == 'dict':
        style = LABELED_DICT_TERM_STYLE

    return f'{style}({term.value}: {term.class_}){Style.RESET_ALL}'


def render_grouped_term(term: Term) -> str:
    values = ', '.join([t.norm_value for t in term.mentions])

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


def log_grouped_terms(grouped_terms: List[Term]):
    non_single_terms = [t for t in grouped_terms if len(t.mentions) > 1]

    term_count = 1
    for term in non_single_terms:
        log_header = '[   GROUP    {: 4}]'.format(term_count)
        term_count += 1

        log(f'{LOG_STYLE}{log_header}{Style.RESET_ALL}: ' + render_grouped_term(term))
        log()


def log_extracted_terms(text: str, terms: List[TermMention]):
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


def log_labeled_terms(text: str, labeled_terms: List[TermMention]):
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


def log_class_predictions(predictions_by_term: Dict[TermMention, List[Tuple[str, float]]]):
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
