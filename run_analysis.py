import json

import inquirer
from colorama import init as colorama_init

from ontology_entities import convert_to_ont_entities
from plot_graph import display_relation_graph
from semantics_analysis.config import load_config, Config
from semantics_analysis.pipelines import *
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_extraction.dict_term_mention_extractor import DictTermExtractor
from semantics_analysis.term_extraction.hybrid_term_mention_extractor import CombinedTermExtractor
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import \
    RobertaTermExtractor
from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ResolveLibraries
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTerms
from semantics_analysis.utils import log_found_relations, AlignedProgress

LOG_STYLE = Style.DIM


def analyze_text(app_config: Config, roberta_term_predictor: Optional[TermMentionExtractor] = None):
    if not roberta_term_predictor:
        roberta_term_predictor = CombinedTermExtractor(
            DictTermExtractor('metadata/terms_by_class.json'),
            RobertaTermExtractor(app_config.device, term_threshold=0.2, class_threshold=0.5)
        )

    llm_term_verifier = LLMTermVerifier()
    llm_term_normalizer = LLMTermNormalizer()

    llm_relation_predictor = LLMRelationExtractor(
        show_explanation=app_config.show_explanation,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses,
        use_all_tokens=True
    )

    text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')
    print()

    with AlignedProgress() as progress:
        llm_reference_resolver = LLMReferenceResolver(
            model=app_config.llm,
            show_explanation=app_config.show_explanation,
            log_prompts=app_config.log_prompts,
            log_llm_responses=app_config.log_llm_responses,
            use_all_tokens=True,
            progress=progress
        )

        semantics_analysis = SequencePipeline(
            Log(message='Predicting terms...'),

            PredictTerms(roberta_term_predictor),

            PreprocessTerms(
                ResolveLibraries(),

                MergeCloseTerms()
            ),

            LogLabeledTerms(),

            VerifyTerms(llm_term_verifier, progress),

            Log(message='Verified terms'),

            LogLabeledTerms(),

            NormalizeTerms(llm_term_normalizer, progress),

            LogNormalizedTerms(),

            ResolveReference(llm_reference_resolver, progress),

            LogGroupedTerms(),

            PredictSemanticRelations(llm_relation_predictor, progress),

            progress=progress
        )

        result = semantics_analysis(AnalysisResult(text))

    objects, ont_relations = convert_to_ont_entities(result.terms, result.relations)

    ont_entities_json = {
        'objects': [o.to_json() for o in objects],
        'relations': [r.to_json() for r in ont_relations]
    }

    with open('ont_entities.json', 'w', encoding='utf-8') as wf:
        json.dump(ont_entities_json, wf, ensure_ascii=False, indent=2)

    if result.relations and app_config.display_graph:
        display_relation_graph(result.terms, result.relations)
    else:
        log_found_relations(result.relations)


def main():
    colorama_init()

    app_config = load_config('config.yml')

    term_predictor = CombinedTermExtractor(
        DictTermExtractor('metadata/terms_by_class.json'),
        RobertaTermExtractor(app_config.device, term_threshold=0.2, class_threshold=0.5)
    )

    while True:
        analyze_text(app_config, roberta_term_predictor=term_predictor)

        question = inquirer.questions.List(
            name='answer',
            message='Продолжить анализировать тексты дальше',
            choices=['Нет.', 'Да.']
        )

        answer = inquirer.prompt([question])['answer']

        if answer == 'Да.':
            continue
        else:
            break


if __name__ == '__main__':
    main()
