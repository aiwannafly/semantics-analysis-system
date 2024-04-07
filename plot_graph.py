import re
from time import sleep
from typing import List

import networkx as nx
from pyvis.network import Network

from semantics_analysis.entities import Relation, read_sentences
from semantics_analysis.term_classification.roberta_term_classifier import LABEL_LIST

colors = [
    '#f44336', '#e81e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688',
    '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722'
]


def add_physics_stop_to_html(filepath):
    with open(filepath, 'r', encoding="utf-8") as file:
        content = file.read()

    # Search for the stabilizationIterationsDone event and insert the network.setOptions line
    pattern = r'(network.once\("stabilizationIterationsDone", function\(\) {)'
    replacement = r'\1\n\t\t\t\t\t\t  // Disable the physics after stabilization is done.\n\t\t\t\t\t\t  network.setOptions({ physics: false });'

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write the modified content back to the file
    with open(filepath, 'w', encoding="utf-8") as file:
        file.write(new_content)


def get_color(class_: str):
    if class_ in LABEL_LIST:
        return colors[LABEL_LIST.index(class_)]
    return '#000000'


def display_relation_graph(relations: List[Relation]):
    if not relations:
        return

    nx_graph = nx.MultiDiGraph()

    mass = 10

    size = 20

    for rel in relations:
        first = f'{rel.term1.value}\n({rel.term1.class_})'
        second = f'{rel.term2.value}\n({rel.term2.class_})'

        color1 = get_color(rel.term1.class_)
        color2 = get_color(rel.term2.class_)

        nx_graph.add_node(first, mass=mass, size=size, label=first, color=color1)
        nx_graph.add_node(second, mass=mass, size=size, label=second, color=color2)

        edge_color = '#' + color1[1:] + 'CC'

        nx_graph.add_edge(first, second, label=rel.predicate, color=edge_color)

    nt = Network(width='100%', height='800px', notebook=False, directed=True)

    nt.from_nx(nx_graph)

    # nt.repulsion(node_distance=300)
    # nt.barnes_hut(gravity=-200)
    nt.force_atlas_2based(gravity=-50, central_gravity=0.03)

    nt.show_buttons(filter_=['physics'])
    nt.show('relations.html', notebook=False)


def main():
    sentences = read_sentences('tests/sentences.json')

    for sent in sentences:
        if len(sent.relations) > 14:
            display_relation_graph(sent.relations)
            sleep(3)


if __name__ == '__main__':
    main()
