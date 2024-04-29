import json
import sys
from typing import List, Tuple, Dict, Optional

import inquirer
import nltk
from colorama import Style
from colorama import init as colorama_init
from rich.progress import Progress

from ontology_entities import convert_to_ont_entities
from parse_habr import Doc
from plot_graph import display_relation_graph
from semantics_analysis.config import load_config
from semantics_analysis.entities import Term, GroupedTerm, Relation
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor
from semantics_analysis.term_classification.hybrid_term_classifier import HybridTermClassifier
from semantics_analysis.term_classification.roberta_term_classifier import RobertaTermClassifier
from semantics_analysis.term_classification.dict_term_classifier import DictTermClassifier
from semantics_analysis.term_classification.term_classifier import TermClassifier
from semantics_analysis.term_extraction.roberta_term_extractor import RobertaTermExtractor
from semantics_analysis.term_extraction.term_extractor import TermExtractor
from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ComputerScienceTermPostProcessor
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTermPostProcessor
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from semantics_analysis.utils import log_class_predictions, log_grouped_terms, log_labeled_terms, log_extracted_terms, \
    log_found_relations, union_groups, normalize_relations, normalize_groups, normalize_term_values


class Result:
    terms: List[GroupedTerm]
    relations: List[Relation]

    def __init__(self, terms: List[GroupedTerm], relations: List[Relation]):
        self.terms = terms
        self.relations = relations


def analyze_paragraph(
        text: str,
        term_extractor: TermExtractor,
        term_classifier: TermClassifier,
        reference_resolver: Optional[ReferenceResolver],
        relation_extractor: RelationExtractor,
        term_postprocessors: List[TermPostProcessor],
        progress: Progress,
        extract_relations: bool = True
) -> Result:
    text = text.replace('�', '').strip()

    sentences = nltk.tokenize.sent_tokenize(text)

    text = ' '.join(sentences)

    result = Result([], [])

    terms = term_extractor(text)

    if not terms:
        return result

    sentence_idx_by_term = {}

    text_offset = 0
    curr_term_idx = 0

    for sent_idx, sent in enumerate(sentences):
        text_offset += len(sent) + 1

        for i in range(curr_term_idx, len(terms)):
            if terms[i].end_pos <= text_offset:
                sentence_idx_by_term[terms[i]] = sent_idx
                curr_term_idx += 1
            else:
                break

    terms_by_sent_idx = {}

    for term, sent_idx in sentence_idx_by_term.items():
        if sent_idx not in terms_by_sent_idx:
            terms_by_sent_idx[sent_idx] = [term]
        else:
            terms_by_sent_idx[sent_idx].append(term)

    predictions_by_term: Dict[Term, List[Tuple[str, float]]] = {}

    labeled_terms = []

    text_offset = 0

    for sent_idx in range(0, len(sentences)):
        sent = sentences[sent_idx]

        if sent_idx not in terms_by_sent_idx:
            text_offset += len(sent) + 1
            continue

        terms = terms_by_sent_idx[sent_idx]

        for term in terms:
            term.start_pos -= text_offset
            term.end_pos -= text_offset

        sent_labeled_terms = term_classifier(sent, terms, predictions_by_term)

        for term_postprocessor in term_postprocessors:
            sent_labeled_terms = term_postprocessor(sent_labeled_terms)

        for term in sent_labeled_terms:
            term.start_pos += text_offset
            term.end_pos += text_offset
            labeled_terms.append(term)

        text_offset += len(sent) + 1

    if not extract_relations:
        grouped_terms = [GroupedTerm(t.class_, [t], normalize=False) for t in labeled_terms]
        return Result(grouped_terms, [])

    if reference_resolver:
        grouped_terms = reference_resolver(labeled_terms, text, progress)
    else:
        grouped_terms = [GroupedTerm(t.class_, [t], normalize=False) for t in labeled_terms]

    result.terms.extend(grouped_terms)

    labeled_terms = [t.as_single() for t in grouped_terms]

    text_and_terms = [(text, labeled_terms)]

    found_relations = []
    for text_part, labeled_terms in text_and_terms:

        relations = relation_extractor(text_part, labeled_terms, progress)

        found_relations = [r for r in relations]

    result.relations.extend(found_relations)
    return result


def analyze_article(
        article_id: int,
        term_extractor: TermExtractor,
        term_classifier: TermClassifier,
        reference_resolver: ReferenceResolver,
        relation_extractor: RelationExtractor,
        term_postprocessors: List[TermPostProcessor],
        term_normalizer: TermNormalizer
) -> Result:
    doc = Doc.from_article(article_id)

    result = Result([], [])

    if len(doc.paragraphs) == 0:
        return result

    with Progress() as progress:
        paragraph_task = progress.add_task(description=f'Paragraph 0/{len(doc.paragraphs)}', total=len(doc.paragraphs))
        count = 1
        for p in doc.paragraphs:
            p_result = analyze_paragraph(
                p,
                term_extractor,
                term_classifier,
                reference_resolver,
                relation_extractor,
                term_postprocessors,
                progress
            )

            result.terms.extend(p_result.terms)
            result.relations.extend(p_result.relations)

            progress.update(paragraph_task, description=f'Paragraph {count}/{len(doc.paragraphs)}', advance=1)
            count += 1

        progress.remove_task(paragraph_task)

    # we need to union term groups from different paragraphs
    result.terms = union_groups(result.terms)
    result.relations = normalize_relations(result.terms, result.relations)
    result.terms = normalize_groups(result.terms)

    print('Normalizing terms...')
    result.terms, result.relations = normalize_term_values(result.terms, result.relations, term_normalizer)

    return result


def main():
    if len(sys.argv) < 2:
        print('usage: analyze_article.py <article-id>')
        exit(0)

    article_id = int(sys.argv[-1])

    app_config = load_config('config.yml')

    term_extractor = RobertaTermExtractor(app_config.device)

    term_postprocessors = [
        ComputerScienceTermPostProcessor(),
        MergeCloseTermPostProcessor()
    ]

    roberta_term_classifier = RobertaTermClassifier(app_config.device)

    if app_config.use_dict:
        term_classifier = HybridTermClassifier(
            DictTermClassifier('metadata/terms_by_class.json'),
            roberta_term_classifier
        )
    else:
        term_classifier = roberta_term_classifier

    relation_extractor = LLMRelationExtractor(
        model=app_config.llm,
        show_explanation=app_config.show_explanation,
        huggingface_hub_token=app_config.huggingface_hub_token,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses,
        use_all_tokens=True
    )

    reference_resolver = LLMReferenceResolver(
        model=app_config.llm,
        show_explanation=app_config.show_explanation,
        huggingface_hub_token=app_config.huggingface_hub_token,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses,
        use_all_tokens=True
    )

    term_normalizer = LLMTermNormalizer()

    result = analyze_article(
        article_id,
        term_extractor,
        term_classifier,
        reference_resolver,
        relation_extractor,
        term_postprocessors,
        term_normalizer
    )

    objects, ont_relations = convert_to_ont_entities(result.terms, result.relations)

    ont_entities_json = {
        'objects': [o.to_json() for o in objects],
        'relations': [r.to_json() for r in ont_relations]
    }

    with open('ont_entities.json', 'w', encoding='utf-8') as wf:
        json.dump(ont_entities_json, wf, ensure_ascii=False, indent=2)

    print('Saved ontology entities in "ont_entities.json".')
    print()

    print('Building relations graph...')
    display_relation_graph(
        result.terms,
        result.relations,
        output_file=f'article_{article_id}.html'
    )


if __name__ == '__main__':
    main()
