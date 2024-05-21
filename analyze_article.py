import json
import sys

from rich.console import Console
from rich.progress import Progress
from rich.theme import Theme

from ontology_entities import convert_to_ont_entities
from parse_habr import Doc
from plot_graph import display_relation_graph
from semantics_analysis.config import load_config, Config
from semantics_analysis.pipelines import AnalysisResult, SequencePipeline, PredictTerms, PreprocessTerms, VerifyTerms, \
    NormalizeTerms, ResolveReference, PredictSemanticRelations
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_extraction.dict_term_mention_extractor import DictTermExtractor
from semantics_analysis.term_extraction.hybrid_term_mention_extractor import CombinedTermExtractor
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import RobertaTermExtractor
from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
from semantics_analysis.term_post_processing.computer_science_term_post_processor import \
    ResolveLibraries
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTerms
from semantics_analysis.utils import union_term_mentions, AlignedProgress


def analyze_article(article_id: int, app_config: Config) -> AnalysisResult:
    doc = Doc.from_article(article_id)

    term_predictor = CombinedTermExtractor(
        DictTermExtractor('metadata/terms_by_class.json'),
        RobertaTermExtractor(app_config.device, term_threshold=0.2, class_threshold=0.5)
    )

    relation_predictor = LLMRelationExtractor(
        show_explanation=app_config.show_explanation,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses,
        use_all_tokens=True
    )

    result = AnalysisResult(f'{article_id}')

    if len(doc.paragraphs) == 0:
        return result

    custom_theme = Theme({"bar.complete": "rgb(206,89,227)"})
    console = Console(theme=custom_theme)

    with AlignedProgress(console=console) as progress:
        paragraph_task = progress.add_task(description=f'Paragraph 0/{len(doc.paragraphs)}', total=len(doc.paragraphs))
        count = 1

        reference_resolver = LLMReferenceResolver(
            model=app_config.llm,
            show_explanation=app_config.show_explanation,
            log_prompts=app_config.log_prompts,
            log_llm_responses=app_config.log_llm_responses,
            use_all_tokens=True,
            progress=progress
        )

        semantics_analysis = SequencePipeline(
            PredictTerms(term_predictor),

            PreprocessTerms(
                ResolveLibraries(),

                MergeCloseTerms()
            ),

            VerifyTerms(LLMTermVerifier(), progress),

            NormalizeTerms(LLMTermNormalizer(), progress),

            ResolveReference(reference_resolver, progress),

            PredictSemanticRelations(relation_predictor, progress),

            progress=progress
        )

        for paragraph in doc.paragraphs:
            p_result = semantics_analysis(AnalysisResult(paragraph))

            result.terms.extend(p_result.terms)
            result.relations.extend(p_result.relations)

            progress.update(paragraph_task, description=f'Paragraph {count}/{len(doc.paragraphs)}', advance=1)
            count += 1

        progress.remove_task(paragraph_task)

        # we need to union term groups from different paragraphs
        result.terms = union_term_mentions(result.terms)
        result.relations = list(set(result.relations))

    return result


def main():
    if len(sys.argv) < 2:
        print('usage: analyze_article.py <article-id>')
        exit(0)

    article_id = int(sys.argv[-1])

    app_config = load_config('config.yml')

    result = analyze_article(article_id, app_config)

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
