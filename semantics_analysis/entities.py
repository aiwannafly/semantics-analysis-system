import json
from typing import Dict, Any, List, Optional


class Term:
    def __init__(self, term_class: str, value: str):
        self.class_ = term_class
        self.value = value

    def to_json(self) -> Dict[str, Any]:
        return {
            'class': self.class_,
            'value': self.value
        }

    @staticmethod
    def parse(term_dict):
        return Term(term_dict['class'], term_dict['value'])

    def __repr__(self):
        return f'Term(class={self.class_}, value={self.value})'

    def __eq__(self, other):
        if isinstance(other, Term):
            return other.value == self.value and other.class_ == self.class_

        return False


class Relation:
    def __init__(self, term1: Term, predicate: str, term2: Term):
        self.term1 = term1
        self.term2 = term2
        self.predicate = predicate

    def to_json(self) -> Dict[str, Any]:
        return {
            'term1': self.term1.to_json(),
            'predicate': self.predicate,
            'term2': self.term2.to_json(),
        }

    @staticmethod
    def parse(rel_dict):
        return Relation(Term.parse(rel_dict['term1']), rel_dict['predicate'], Term.parse(rel_dict['term2']))

    def __repr__(self):
        return f'Relation(term1={self.term1}, predicate={self.predicate}, term2={self.term2})'


class Sentence:
    def __init__(self,
                 sent_id: int,
                 text: str,
                 terms: Dict[str, List[str]],
                 relations: List[Relation]):
        self.id = sent_id
        self.text = text
        self.terms = terms
        self.relations = relations

    @staticmethod
    def parse(sent_dict):
        sent_id = sent_dict['sent_id']
        text = sent_dict['text']
        terms = sent_dict['terms']
        relations = [Relation.parse(r) for r in sent_dict['relations']]

        return Sentence(sent_id, text, terms, relations)

    def find_relation_by_id(self, relation_id: str) -> Optional[Relation]:
        for r in self.relations:
            if f'{r.term1.class_}_{r.predicate}_{r.term2.class_}' == relation_id:
                return r
        return None

    def find_term_by_class_id(self, class_id: str) -> Optional[str]:
        if class_id not in self.terms:
            return None

        return self.terms[class_id][0]

    def to_json(self):
        return {
            'sent_id': self.id,
            'text': self.text,
            'terms': self.terms,
            'relations': [r.to_json() for r in self.relations]
        }

    def __repr__(self):
        return f'Sentence(id={self.id}, text="{self.text}", terms={self.terms}, relations={self.relations})'


def main():
    rel = Relation(
        Term('Method', 'AdaGRAD'),
        'solves',
        Term('Task', 'Classification')
    )

    rel_json = json.dumps(rel.to_json())

    print(rel_json)


if __name__ == '__main__':
    main()
