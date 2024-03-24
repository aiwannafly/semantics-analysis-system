from typing import List

import networkx as nx
import matplotlib.pyplot as plt

from semantics_analysis.entities import Relation


def draw_labeled_multigraph(G, ax=None):
    font_size = 12

    pos = nx.shell_layout(G)

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        ax=ax,
        clip_on=False
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=(.3, .5, .2, .2),
        arrowsize=40,
        # arrowstyle='fancy',
        ax=ax
    )

    labels = {
        (edge[0], edge[1]): f"{attrs['predicate']}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        font_color="purple",
        font_size=font_size,
        bbox={"alpha": 0},
        ax=ax,
    )


def display_relation_graph(relations: List[Relation]):
    if not relations:
        return

    G = nx.MultiDiGraph()

    for rel in relations:
        first = f'{rel.term1.value}\n({rel.term1.class_})'
        second = f'{rel.term2.value}\n({rel.term2.class_})'

        G.add_edge(first, second, predicate=rel.predicate)

    fig, ax = plt.subplots(1, 1)

    draw_labeled_multigraph(G, ax)

    fig.set_figwidth(12)
    fig.set_figheight(8)
    ax.axis('off')
    ax.set_title('Relations graph')

    plt.show()


# sentences = read_sentences('tests/sentences.json')
#
# for sent in sentences:
#     if len(sent.relations) > 5:
#         display_relation_graph(sent.relations)
