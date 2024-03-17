import json
import termcharts
from rich import print
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

with open('tests/scores.json', 'r', encoding='utf-8') as f:
    scores_by_relation = json.load(f)

groups = []
curr_group = []

avg_recall = 0
avg_precision = 0
count = 0

for relation, scores in scores_by_relation.items():
    correct = scores['predicted']['correct']['count']
    incorrect = scores['predicted']['incorrect']['count']

    if correct + incorrect > 0:
        precision = correct / (correct + incorrect)
    else:
        precision = 0

    found = scores['expected']['found']['count']
    not_found = scores['expected']['not_found']['count']

    if found + not_found > 0:
        recall = found / (found + not_found)
    else:
        recall = 0

    if recall == 0 or precision == 0:
        continue

    recall = int(recall * 100) / 100.0
    precision = int(precision * 100) / 100.0

    avg_recall += recall
    avg_precision += precision
    count += 1

    occurrences_count = max(found + not_found, correct + incorrect)

    curr_group.append((relation, occurrences_count, {'Recall': recall, 'Precision': precision}))

    if len(curr_group) == 4:
        groups.append(curr_group)
        curr_group = []

if curr_group:
    while len(curr_group) < 4:
        curr_group.append(None)
    groups.append(curr_group)

layouts = []
for group in groups:
    rendered = []

    for score in group:
        if score:
            title, occurrences_count, scores = score

            title = f'{occurrences_count} {title}'
            rendered.append(Panel(termcharts.bar(scores, title=title, rich=True)))
        else:
            rendered.append(Panel(Text('Empty cell')))

    layout = Layout()
    layout.split_column(Layout(name="upper"), Layout(name="lower"))
    layout["upper"].split_row(
        Layout(name="uleft"),
        Layout(name="uright"),
    )
    layout["lower"].split_row(
        Layout(name="lleft"),
        Layout(name="lright"),
    )
    layout["uleft"].update(rendered[0])
    layout["uright"].update(rendered[1])
    layout["lleft"].update(rendered[2])
    layout["lright"].update(rendered[3])

    layouts.append(layout)
    print(layout)

print('Average recall: {:.2f}'.format(avg_recall / count))
print('Average precision: {:.2f}'.format(avg_precision / count))
