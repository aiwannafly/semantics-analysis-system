import json
import os
import sys
from time import sleep
from typing import List, Dict, Any, Set

from huggingface_hub.utils import HfHubHTTPError

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence, Relation
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extractor.ontology_utils import (predicates_by_class_pair, is_symmetric,
                                                                  loaded_relation_ids)
from rich.progress import Progress

from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor


def update_scores(
        sent: Sentence,
        predicted_relations: Set[Relation],
        expected_relations: Set[Relation],
        ignored_relations: Set[Relation],
        scores: Dict[str, Any]
):
    temp = predicted_relations

    predicted_relations = set()

    for rel in temp:
        if rel.id not in loaded_relation_ids:
            ignored_relations.add(rel.id)
            continue

        if is_symmetric[rel.id]:
            inverse_rel = rel.inverse()
            if rel not in expected_relations and inverse_rel in expected_relations:
                predicted_relations.add(inverse_rel)
            else:
                predicted_relations.add(rel)
        else:
            predicted_relations.add(rel)

    for rel in expected_relations:
        if rel.id not in scores:
            ignored_relations.add(rel.id)
            continue

        if rel in predicted_relations:
            scores[rel.id]['expected']['found']['count'] += 1
            scores[rel.id]['expected']['found']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})
        else:
            scores[rel.id]['expected']['not_found']['count'] += 1
            scores[rel.id]['expected']['not_found']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})

    for rel in predicted_relations:
        if rel.id not in scores:
            ignored_relations.add(rel.id)
            continue

        if rel in expected_relations:
            scores[rel.id]['predicted']['correct']['count'] += 1
            scores[rel.id]['predicted']['correct']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})
        else:
            scores[rel.id]['predicted']['incorrect']['count'] += 1
            scores[rel.id]['predicted']['incorrect']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})


def calculate_scores(
        scores: Dict[str, Any],
        relation_extractor: RelationExtractor,
        sentences_to_check: List[Sentence],
        last_sent_id: int
) -> (int, bool):
    for (class1, class2), predicates in predicates_by_class_pair.items():
        for predicate in predicates:
            rel_id = f'{class1}_{predicate}_{class2}'

            if rel_id in scores:
                continue

            scores[rel_id] = {
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

    ignored_relations = set()

    counter = 1

    total_sentences = len(sentences_to_check)

    with Progress() as progress:
        sentence_task = progress.add_task(
            description=f'[green]Sentence {counter}/{total_sentences}',
            total=total_sentences
        )

        for sent in sentences_to_check:
            if sent.id <= last_sent_id:
                counter += 1
                progress.update(sentence_task, advance=1, description=f'[green]Sentence {counter}/{total_sentences}')
                continue

            expected_relations = set()

            #  we consider only those relations, that are actually covered at the current moment
            for rel in sent.relations:
                if rel.id in loaded_relation_ids:
                    expected_relations.add(rel)

            if not expected_relations:
                counter += 1
                progress.update(sentence_task, advance=1, description=f'[green]Sentence {counter}/{total_sentences}')
                continue

            try:
                term_pairs = relation_extractor.get_pairs_to_consider(sent.terms)

                predicted_relations = set()

                total_pairs = len(term_pairs)
                pair_count = 1

                extract_rel_task = progress.add_task(
                    description=f'[cyan]Term pair {pair_count}/{total_pairs}',
                    total=total_pairs
                )

                progress.update(
                    extract_rel_task,
                    description=f'[cyan]Term pair {pair_count}/{total_pairs}',
                    advance=1
                )

                for rel in relation_extractor.analyze_term_pairs(sent.text, term_pairs):
                    pair_count += 1

                    if rel:
                        predicted_relations.add(rel)
                    progress.update(
                        extract_rel_task,
                        description=f'[cyan]Term pair {pair_count}/{total_pairs}',
                        advance=1
                    )

                progress.remove_task(extract_rel_task)

            except HfHubHTTPError as e:
                if 'Rate limit reached' in e.server_message:
                    print('[DETECTED TOKEN LIMIT]')
                    break
                else:
                    print(e)
                    break
            except Exception as e:
                print(e)
                return last_sent_id, False

            update_scores(sent, predicted_relations, expected_relations, ignored_relations, scores)

            counter += 1
            progress.update(sentence_task, advance=1, description=f'[green]Sentence {counter}/{total_sentences}')
            last_sent_id = sent.id

    if ignored_relations:
        print(f'Ignored these relations: {ignored_relations}')

    return last_sent_id, counter >= total_sentences


def main():
    relations_to_consider = []

    for arg in sys.argv[1:]:
        if arg in loaded_relation_ids:
            relations_to_consider.append(arg)
        else:
            print(f'Invalid relation id: {arg}')
            return 0

    config = load_config('config.yml')

    sentences = read_sentences('tests/sentences.json')

    if not relations_to_consider:
        sentences_to_check = sentences
    else:
        sentences_to_check = []

        class_pairs_to_consider = []
        for rel in relations_to_consider:
            class1, _, class2 = rel.split('_')

            if (class1, class2) not in class_pairs_to_consider:
                class_pairs_to_consider.append((class1, class2))

        for sent in sentences:
            for class1, class2 in class_pairs_to_consider:
                if sent.find_term_by_class_id(class1) and sent.find_term_by_class_id(class2):
                    sentences_to_check.append(sent)
                    break


    if relations_to_consider:
        total_id = '_and_'.join(relations_to_consider)

        if not os.path.exists(f'tests/{total_id}'):
            os.makedirs(f'tests/{total_id}')

        lock_path = f'tests/{total_id}/last_stop.lock'
        scores_path = f'tests/{total_id}/scores.json'
    else:
        lock_path = 'tests/last_stop.lock'
        scores_path = 'tests/scores.json'

    try:
        with open(lock_path, 'r', encoding='utf-8') as f:
            last_sent_id = int(f.readline().strip())

        with open(scores_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
    except Exception:
        scores = {}
        last_sent_id = 0

    tokens = [
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',   
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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',
        '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'
    ]

    token_idx = 3
    while True:
        prev_last_sent_id = last_sent_id

        relation_extractor = LLMRelationExtractor(
            model=config.llm,
            huggingface_hub_token=tokens[token_idx]
        )

        last_sent_id, finished = calculate_scores(scores, relation_extractor, sentences_to_check, last_sent_id)

        if prev_last_sent_id == last_sent_id:
            break
        else:
            token_idx = (token_idx + 1) % len(tokens)
            with open(lock_path, 'w', encoding='utf-8') as f:
                f.write(f'{last_sent_id}')

            with open(scores_path, 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)

            if finished:
                print('Scores were saved.')
                break

            sleep_time_secs = 5
            print(f'Scores were updated. Sleep for {sleep_time_secs} secs until new attempt.')
            sleep(sleep_time_secs)


if __name__ == '__main__':
    main()
