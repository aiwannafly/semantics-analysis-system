import json
from typing import List, Dict, Any

from huggingface_hub.utils import HfHubHTTPError

from semantics_analysis.entities import read_sentences, Sentence
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extractor.ontology_utils import predicates_by_class_pair
from tqdm import tqdm

from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor


def calculate_scores(
        relation_extractor: RelationExtractor,
        sentences: List[Sentence]
) -> (Dict[str, Any], int):
    scores = {}

    sentences_to_check = []

    for sent in sentences:
        has_expected_relations = False

        #  we consider only those relations, that are actually covered at the current moment
        for rel in sent.relations:
            if (rel.term1.class_, rel.term2.class_) in predicates_by_class_pair:
                has_expected_relations = True
                break

        if has_expected_relations:
            sentences_to_check.append(sent)

    for (class1, class2), predicates in predicates_by_class_pair.items():
        for predicate in predicates:
            scores[f'{class1}_{predicate}_{class2}'] = {
                'expected': {
                    'found': {
                        'count': 0,
                        'examples': []
                    },
                    'not_found': {
                        'count': 0,
                        'examples': []
                    }
                },
                'predicted': {
                    'correct': {
                        'count': 0,
                        'examples': []
                    },
                    'incorrect': {
                        'count': 0,
                        'examples': []
                    }
                },
            }

    tqdm_loop = tqdm(sentences_to_check, colour='green')

    ignored_relations = set()

    counter = 1
    for sent in tqdm_loop:
        tqdm_loop.set_description(desc=f'Sentence {counter}/{len(sentences_to_check)}')

        expected_relations = set()

        #  we consider only those relations, that are actually covered at the current moment
        for rel in sent.relations:
            if (rel.term1.class_, rel.term2.class_) in predicates_by_class_pair:
                expected_relations.add(rel)

        if not expected_relations:
            continue

        try:
            predicted_relations = set(relation_extractor(sent.text, sent.terms))
        except HfHubHTTPError as e:
            if 'Rate limit reached' in e.server_message:
                # here we can add token substituion
                print('[DETECTED LIMIT EXCEEDING]')
            break
        except Exception:
            break

        for rel in expected_relations:
            if rel.get_id() not in scores:
                ignored_relations.add(rel.get_id())
                continue

            if rel in predicted_relations:
                scores[rel.get_id()]['expected']['found']['count'] += 1
                scores[rel.get_id()]['expected']['found']['examples'].append({'text': sent.text, 'relation': rel.as_str()})
            else:
                scores[rel.get_id()]['expected']['not_found']['count'] += 1
                scores[rel.get_id()]['expected']['not_found']['examples'].append({'text': sent.text, 'relation': rel.as_str()})

        for rel in predicted_relations:
            if rel.get_id() not in scores:
                ignored_relations.add(rel.get_id())
                continue

            if rel in expected_relations:
                scores[rel.get_id()]['predicted']['correct']['count'] += 1
                scores[rel.get_id()]['predicted']['correct']['examples'].append({'text': sent.text, 'relation': rel.as_str()})
            else:
                scores[rel.get_id()]['predicted']['incorrect']['count'] += 1
                scores[rel.get_id()]['predicted']['incorrect']['examples'].append({'text': sent.text, 'relation': rel.as_str()})
        counter += 1

    if ignored_relations:
        print(f'Ignored these relations: {ignored_relations}')

    return scores, counter - 1


def main():
    sentences = read_sentences('tests/sentences.json')

    relation_extractor = LLMRelationExtractor(
        prompt_template_path='prompts/relation_extraction.txt',
        huggingface_hub_token='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'
    )

    scores, count = calculate_scores(relation_extractor, sentences[:200])

    if count == 0:
        return

    with open('tests/scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print('Scores are saved at "tests/scores.json".')


if __name__ == '__main__':
    main()
