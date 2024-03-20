import json

predicates_by_class_pair = {}

prompt_metadata_by_class_pair = {}

is_symmetric = {}

loaded_relation_ids = set()

with open('metadata/relations.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

for key, predicates in metadata.items():
    class1, class2 = key.split('_')

    predicates_by_class_pair[(class1, class2)] = list(predicates.keys())

    prompt_metadata_by_class_pair[(class1, class2)] = predicates

    for predicate, metadata in predicates.items():
        rel_id = f'{class1}_{predicate}_{class2}'

        loaded_relation_ids.add(rel_id)

        if 'symmetric' in metadata and metadata['symmetric']:
            is_symmetric[rel_id] = True
        else:
            is_symmetric[rel_id] = False
