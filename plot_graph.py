import re
from time import sleep
from typing import List

from pyvis.network import Network

from semantics_analysis.entities import Relation, read_sentences
from semantics_analysis.term_classification.roberta_term_classifier import LABEL_LIST

colors = [
    '#f44336', '#e81e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#03a9f4', '#00bcd4', '#009688',
    '#4caf50', '#8bc34a', '#cddc39', '#ff5722', '#ffc107', '#ff9800', '#ffeb3b'
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


def image_by_class(class_: str):
    return f'icons/{class_.lower()}.png'


def preprocess_is_alternative_name(relations: List[Relation]) -> List[Relation]:
    alt_rel = None

    for rel in relations:
        if rel.predicate == 'isAlternativeNameFor':
            alt_rel = rel
            break

    if alt_rel is None:
        return relations

    new_relations = [Relation(alt_rel.term1, 'alternativeNameFor', alt_rel.term2)]
    alt_term = alt_rel.term1

    for rel in relations:
        if rel.term1 != alt_term and rel.term2 != alt_term:
            new_relations.append(rel)

    return preprocess_is_alternative_name(new_relations)


def display_relation_graph(relations: List[Relation]):
    if not relations:
        return

    nt = Network(width='100%', height='800px', notebook=False, directed=True)

    mass = 5

    size = 20

    for rel in relations:
        color1 = get_color(rel.term1.class_)
        color2 = get_color(rel.term2.class_)

        node1 = rel.term1.value
        node2 = rel.term2.value

        title1 = rel.term1.class_
        title2 = rel.term2.class_

        nt.add_node(
            node1,
            title=title1,
            mass=mass,
            size=size,
            label=node1,
            color=color1,
            shape='image',
            image=image_by_class(rel.term1.class_)
        )
        nt.add_node(
            node2,
            title=title2,
            mass=mass,
            size=size,
            label=node2,
            color=color2,
            shape='image',
            image=image_by_class(rel.term2.class_)
        )

        edge_color = '#' + color1[1:] + '77'

        nt.add_edge(
            node1,
            node2,
            label=rel.predicate,
            title=rel.predicate,
            color=edge_color,
            arrowStrikethrough=False,
            font={
                'size': 10,
                'align': 'top'
            }
        )

    # nt.repulsion(node_distance=300)
    # nt.barnes_hut(gravity=-200)
    nt.force_atlas_2based(gravity=-50, central_gravity=0.02, spring_strength=0.09)

    nt.show_buttons(filter_=['physics'])
    nt.show('relations.html', notebook=False)


def main():
    sentences = read_sentences('tests/sentences.json')

    for sent in sentences:
        if len(sent.relations) > 14:
            display_relation_graph(preprocess_is_alternative_name(sent.relations))
            sleep(3)
            # break


if __name__ == '__main__':
    main()
