import json
from typing import List, Dict, Any

from huggingface_hub.utils import HfHubHTTPError

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence
from semantics_analysis.relation_extractor.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extractor.ontology_utils import predicates_by_class_pair
from tqdm import tqdm

from semantics_analysis.relation_extractor.relation_extractor import RelationExtractor

TOKENS = [
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


def calculate_scores(
        scores: Dict[str, Any],
        relation_extractor: RelationExtractor,
        sentences: List[Sentence],
        last_sent_id: int
) -> int:
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

    # def get_next_token() -> Optional[str]:
    #     tqdm_tokens = tqdm(TOKENS, colour='red')
    #
    #     for idx, token_ in enumerate(tqdm_tokens):
    #         tqdm_tokens.set_description(desc=f'Token {idx + 1}/{len(TOKENS)}')
    #         yield token_
    #
    #     yield None

    ignored_relations = set()

    # relation_extractor = LLMRelationExtractor(
    #     huggingface_hub_token=get_next_token(),
    #     prompt_template_path='prompts/relation_extraction.txt'
    # )

    tqdm_sentences = tqdm(sentences_to_check, colour='green', leave=False)

    counter = 1
    for sent in tqdm_sentences:
        tqdm_sentences.set_description(desc=f'Sentence {counter}/{len(sentences_to_check)}')

        tqdm_sentences.set_postfix({'sent_id': sent.id})

        if sent.id <= last_sent_id:
            counter += 1
            continue

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
                print('[DETECTED TOKEN LIMIT]')
                break
            else:
                print(e)
                break
        except Exception as e:
            print(e)
            return last_sent_id

        for rel in expected_relations:
            if rel.get_id() not in scores:
                ignored_relations.add(rel.get_id())
                continue

            if rel in predicted_relations:
                scores[rel.get_id()]['expected']['found']['count'] += 1
                scores[rel.get_id()]['expected']['found']['examples'].append(
                    {'text': sent.text, 'relation': rel.as_str()})
            else:
                scores[rel.get_id()]['expected']['not_found']['count'] += 1
                scores[rel.get_id()]['expected']['not_found']['examples'].append(
                    {'text': sent.text, 'relation': rel.as_str()})

        for rel in predicted_relations:
            if rel.get_id() not in scores:
                ignored_relations.add(rel.get_id())
                continue

            if rel in expected_relations:
                scores[rel.get_id()]['predicted']['correct']['count'] += 1
                scores[rel.get_id()]['predicted']['correct']['examples'].append(
                    {'text': sent.text, 'relation': rel.as_str()})
            else:
                scores[rel.get_id()]['predicted']['incorrect']['count'] += 1
                scores[rel.get_id()]['predicted']['incorrect']['examples'].append(
                    {'text': sent.text, 'relation': rel.as_str()})
        counter += 1
        last_sent_id = sent.id

    if ignored_relations:
        print(f'Ignored these relations: {ignored_relations}')

    return last_sent_id


def main():
    config = load_config('config.yml')

    sentences = read_sentences('tests/sentences.json')

    try:
        with open('tests/last_stop.lock', 'r', encoding='utf-8') as f:
            last_sent_id = int(f.readline().strip())

        with open('tests/scores.json', 'r', encoding='utf-8') as f:
            scores = json.load(f)
    except Exception:
        scores = {}
        last_sent_id = 0

    relation_extractor = LLMRelationExtractor(
        model=config.llm,
        prompt_template_path='prompts/relation_extraction.txt',
        huggingface_hub_token='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'
    )

    prev_last_sent_id = last_sent_id

    last_sent_id = calculate_scores(scores, relation_extractor, sentences, last_sent_id)

    if prev_last_sent_id == last_sent_id:
        return

    with open('tests/last_stop.lock', 'w', encoding='utf-8') as f:
        f.write(f'{last_sent_id}')

    with open('tests/scores.json', 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print('Scores are saved at "tests/scores.json".')


if __name__ == '__main__':
    main()
