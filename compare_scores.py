import sys

from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from rich.text import Text

from display_scores import parse_scores

if len(sys.argv) < 1 + 2:
    print('usage: compare_scores.py <scores1>.json <scores2>.json')
    exit(0)

scores1_path = sys.argv[-2]
scores2_path = sys.argv[-1]

storage1 = {}
storage2 = {}

results1 = parse_scores(scores1_path, None, storage1)
results2 = parse_scores(scores2_path, None, storage2)

table = Table(title="Scores comparison")

table.add_column(header="Relation", justify="left", no_wrap=True)
table.add_column(header="Recall 1")
table.add_column(header="Recall 2")
table.add_column(header="Precision 1")
table.add_column(header="Precision 2")


scores_by_rel = {}

for rel, _, scores in results1:
    scores_by_rel[rel] = [scores]

for rel, _, scores in results2:
    if rel not in scores_by_rel:
        print(f'Ignored {rel}')
        continue
    scores_by_rel[rel].append(scores)

BETTER_STYLE = 'green'
WORSE_STYLE = 'red'
CONFLICT_STYLE = 'yellow'

for rel, results in scores_by_rel.items():
    if len(results) == 1:
        print(f'Ignored {rel}')
        continue

    scores1, scores2 = results

    recall1, recall2 = scores1['Recall'], scores2['Recall']

    if recall2 < recall1:
        recall2_style = WORSE_STYLE
    elif recall2 > recall1:
        recall2_style = BETTER_STYLE
    else:
        recall2_style = None

    precision1, precision2 = scores1['Precision'], scores2['Precision']

    if precision2 < precision1:
        precision2_style = WORSE_STYLE
    elif precision2 > precision1:
        precision2_style = BETTER_STYLE
    else:
        precision2_style = None

    if not precision2_style and not recall2_style:
        common_style = None
    elif precision2_style and recall2_style:
        styles = {precision2_style, recall2_style}

        if len(styles) > 1:
            common_style = CONFLICT_STYLE
        elif WORSE_STYLE in styles:
            common_style = WORSE_STYLE
        else:
            common_style = BETTER_STYLE
    elif precision2_style or recall2_style:
        styles = {precision2_style, recall2_style}

        if WORSE_STYLE in styles:
            common_style = WORSE_STYLE
        else:
            common_style = BETTER_STYLE
    else:
        common_style = None

    table.add_row(
        Text(rel, style=common_style),
        Text(str(recall1)),
        Text(str(recall2), style=recall2_style),
        Text(str(precision1)),
        Text(str(precision2), style=precision2_style)
    )

console = Console()
console.print(table)
