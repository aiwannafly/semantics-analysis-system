from typing import List

import networkx as nx
from pyvis.network import Network

from semantics_analysis.entities import Relation, read_sentences
from semantics_analysis.term_classification.roberta_term_classifier import LABEL_LIST

colors = [
    '#f44336', '#e81e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688',
    '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722'
]


def display_relation_graph(relations: List[Relation]):
    if not relations:
        return

    nx_graph = nx.MultiDiGraph()

    for rel in relations:
        first = f'{rel.term1.value}\n({rel.term1.class_})'
        second = f'{rel.term2.value}\n({rel.term2.class_})'

        mass = 7
        color1 = colors[LABEL_LIST.index(rel.term1.class_)]
        color2 = colors[LABEL_LIST.index(rel.term2.class_)]

        size = 20

        nx_graph.add_node(first, mass=mass, size=size, label=first, color=color1)
        nx_graph.add_node(second, mass=mass, size=size, label=second, color=color2)

        nx_graph.add_edge(first, second, label=rel.predicate)

    nt = Network(width='100%', height='800px', notebook=False, directed=True)

    nt.from_nx(nx_graph)

    nt.show('relations.html', notebook=False)


# sentences = read_sentences('tests/sentences.json')
#
# for sent in sentences:
#     if len(sent.relations) > 5:
#         display_relation_graph(sent.relations)
#         break
