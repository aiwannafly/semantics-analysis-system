import json
import os
import sys
from time import sleep
from typing import List, Dict, Any, Set, Optional

from huggingface_hub.utils import HfHubHTTPError

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence, Relation
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extraction.ontology_utils import predicates_by_class_pair, loaded_relation_ids
from rich.progress import Progress

from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor


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
        reference_resolver: ReferenceResolver,
        sentences_to_check: List[Sentence],
        last_sent_id: int,
        progress: Progress = Progress(),
        relation_to_consider: Optional[str] = None
) -> (int, bool):
    for (class1, class2), predicates in predicates_by_class_pair.items():
        for predicate in predicates:
            rel_id = f'{class1}_{predicate}_{class2}'

            if rel_id in scores:
                continue

            scores[rel_id] = {
                'predicted': {
                    'incorrect': {
                        'count': 0,
                        'examples': []
                    },
                    'correct': {
                        'count': 0,
                        'examples': []
                    }
                },
                'expected': {
                    'not_found': {
                        'count': 0,
                        'examples': []
                    },
                    'found': {
                        'count': 0,
                        'examples': []
                    }
                }
            }

    ignored_relations = set()

    counter = 1

    total_sentences = len(sentences_to_check)

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

        group_task = progress.add_task(description='Grouping terms')

        try:
            grouped_terms = reference_resolver(sent.terms, sent.text)
        except Exception:
            progress.remove_task(group_task)
            progress.remove_task(sentence_task)
            return last_sent_id, False

        progress.remove_task(group_task)

        sent_terms = [t.as_single() for t in grouped_terms]

        term_pairs = relation_extractor.get_pairs_to_consider(sent_terms)

        if relation_to_consider:
            class1, _, class2 = relation_to_consider.split('_')

            term_pairs = [pair for pair in term_pairs if pair[0].class_ == class1 and pair[1].class_ == class2]

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
        try:
            relations = relation_extractor.analyze_term_pairs(sent.text, term_pairs)

            for rel in relations:
                pair_count += 1

                if rel:
                    predicted_relations.add(rel)

                progress.update(
                    extract_rel_task,
                    description=f'[cyan]Term pair {pair_count}/{total_pairs}',
                    advance=1
                )
        except Exception:
            progress.remove_task(sentence_task)
            progress.remove_task(extract_rel_task)
            return last_sent_id, False

        progress.remove_task(extract_rel_task)

        update_scores(sent, predicted_relations, expected_relations, ignored_relations, scores)

        counter += 1
        progress.update(sentence_task, advance=1, description=f'[green]Sentence {counter}/{total_sentences}')
        last_sent_id = sent.id

    if counter < total_sentences:
        progress.remove_task(sentence_task)

    return last_sent_id, counter >= total_sentences


def main():
    relation_to_consider = None

    if len(sys.argv) > 1:
        relation_to_consider = sys.argv[1]

        if relation_to_consider not in loaded_relation_ids:
            print(f'Invalid relation id: {relation_to_consider}')
            return 0

    config = load_config('config.yml')

    sentences = read_sentences('tests/sentences.json')

    if not relation_to_consider:
        sentences_to_check = sentences
    else:
        sentences_to_check = []

        class1, _, class2 = relation_to_consider.split('_')

        for sent in sentences:
            if class1 == class2 and len(sent.find_terms_by_class(class1)) >= 2:
                sentences_to_check.append(sent)
                continue

            if class1 != class2 and sent.find_term_by_class(class1) and sent.find_term_by_class(class2):
                sentences_to_check.append(sent)
                continue

    if relation_to_consider:
        if not os.path.exists(f'tests/{relation_to_consider}'):
            os.makedirs(f'tests/{relation_to_consider}')

        lock_path = f'tests/{relation_to_consider}/last_stop.lock'
        scores_path = f'tests/{relation_to_consider}/scores.json'
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

    token_idx = 0
    no_move_count = 0

    with Progress() as progress:
        token_task = progress.add_task(
            total=len(tokens),
            description=f'Token {token_idx}/{len(tokens)}',
            completed=token_idx
        )

        while True:
            prev_last_sent_id = last_sent_id

            relation_extractor = LLMRelationExtractor(
                model=config.llm,
                huggingface_hub_token=tokens[token_idx]
            )

            reference_resolver = LLMReferenceResolver(
                model=config.llm,
                huggingface_hub_token=tokens[token_idx]
            )

            last_sent_id, finished = calculate_scores(
                scores,
                relation_extractor,
                reference_resolver,
                sentences_to_check,
                last_sent_id,
                progress=progress,
                relation_to_consider=relation_to_consider
            )

            if finished:
                with open(lock_path, 'w', encoding='utf-8') as f:
                    f.write(f'{last_sent_id}')

                with open(scores_path, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                print('Scores were saved.')
                break

            token_idx = (token_idx + 1) % len(tokens)
            progress.update(token_task, description=f'Token {token_idx}/{len(tokens)}', completed=token_idx)

            if prev_last_sent_id == last_sent_id:
                no_move_count += 1

                if no_move_count >= len(tokens):
                    print('No tokens are available.')
                    break
            else:
                no_move_count = 0
                with open(lock_path, 'w', encoding='utf-8') as f:
                    f.write(f'{last_sent_id}')

                with open(scores_path, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)

                sleep_time_secs = 5

                wait_task = progress.add_task(description='Token reset...', total=sleep_time_secs, completed=1)

                for i in range(sleep_time_secs):
                    sleep(1)
                    progress.update(wait_task, description='Token reset...', advance=1)
                progress.remove_task(wait_task)


if __name__ == '__main__':
    main()
