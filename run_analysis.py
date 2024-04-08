from typing import List, Tuple, Dict

import inquirer
import nltk
from colorama import Style
from colorama import init as colorama_init
from rich.progress import Progress

from plot_graph import display_relation_graph
from semantics_analysis.config import load_config
from semantics_analysis.entities import Term
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
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ComputerScienceTermPostProcessor
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTermPostProcessor
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from semantics_analysis.utils import log_class_predictions, log_grouped_terms, log_labeled_terms, log_extracted_terms, \
    log_found_relations
from spinner import Spinner

LOG_STYLE = Style.DIM


def analyze_text(
        text: str,
        display_graph: bool,
        show_class_predictions: bool,
        split_on_sentences: bool,
        term_extractor: TermExtractor,
        term_classifier: TermClassifier,
        reference_resolver: ReferenceResolver,
        relation_extractor: RelationExtractor,
        term_postprocessors: List[TermPostProcessor],
):
    text = text.replace('�', '').strip()

    sentences = nltk.tokenize.sent_tokenize(text)

    text = ' '.join(sentences)

    with Spinner():
        terms = term_extractor(text)

    log_extracted_terms(text, terms)

    if not terms:
        return

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

    with Spinner():
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

    log_labeled_terms(text, labeled_terms)

    if show_class_predictions:
        log_class_predictions(predictions_by_term)

    with Progress() as progress:
        grouped_terms = reference_resolver(labeled_terms, text, progress)

    log_grouped_terms(grouped_terms)

    labeled_terms = [t.as_single() for t in grouped_terms]

    if split_on_sentences:
        text_and_terms = []

        for text_part in sentences:
            start_pos = text.index(text_part)
            end_pos = start_pos + len(text_part)

            part_terms = [t for t in labeled_terms if t.start_pos >= start_pos and t.end_pos <= end_pos]

            text_and_terms.append((text_part, part_terms))
    else:
        text_and_terms = [(text, labeled_terms)]

    found_relations = []
    for text_part, labeled_terms in text_and_terms:

        with Progress() as progress:
            relations = relation_extractor(text_part, labeled_terms, progress)

            found_relations = [r for r in relations]

    if found_relations and display_graph:
        display_relation_graph(grouped_terms, found_relations)
    else:
        log_found_relations(found_relations)


def main():
    colorama_init()

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

    while True:
        text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')
        print()

        analyze_text(
            text,
            app_config.display_graph,
            app_config.show_class_predictions,
            app_config.split_on_sentences,
            term_extractor,
            term_classifier,
            reference_resolver,
            relation_extractor,
            term_postprocessors,
        )

        question = inquirer.questions.List(
            name='answer',
            message='Продолжить анализировать тексты дальше?',
            choices=['Нет.', 'Да.']
        )

        answer = inquirer.prompt([question])['answer']

        if answer == 'Да.':
            continue
        else:
            break


if __name__ == '__main__':
    main()
