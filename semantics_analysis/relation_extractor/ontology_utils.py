import json

predicates_by_class_pair = {}

prompt_metadata_by_class_pair = {}

with open('metadata/relations.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

for key, predicates in metadata.items():
    class1, class2 = key.split('_')

    predicates_by_class_pair[(class1, class2)] = list(predicates.keys())

    prompt_metadata_by_class_pair[(class1, class2)] = predicates
