import json
from typing import Dict, List, Optional, Any
from pymorphy3 import MorphAnalyzer


ru_morph = MorphAnalyzer(lang='ru')


class Term:
    def __init__(
            self,
            value: str,
            end_pos: int,
            text: str
    ):
        self.value = value
        self.start_pos = end_pos - len(value)
        self.end_pos = end_pos
        self.text = text

    def __repr__(self):
        return f'Term(value={self.value}, start_pos={self.start_pos}, end_pos={self.end_pos})'

    def __eq__(self, other):
        if isinstance(other, Term):
            return other.value == self.value and other.start_pos == self.start_pos

        return False

    def __hash__(self):
        return hash((self.value, self.start_pos))


class ClassifiedTerm(Term):

    def __init__(
            self,
            term_class: str,
            value: str,
            end_pos: int,
            text: str,
            source: str = 'model'
    ):
        super().__init__(value, end_pos, text)
        self.class_ = term_class
        self.source = source

    @classmethod
    def from_term(cls, term_class: str, term: Term, source: str = 'model'):
        return cls(term_class, term.value, term.end_pos, term.text, source)

    def to_json(self):
        return {
            'class': self.class_,
            'value': self.value,
            'start_pos': self.start_pos
        }

    @staticmethod
    def from_json(term_json: Dict[str, Any], text: str = ''):
        start_pos = term_json['start_pos']
        value = term_json['value']
        class_ = term_json['class']

        end_pos = start_pos + len(value)

        return ClassifiedTerm(class_, value, end_pos, text)

    def __repr__(self):
        return f'ClassifiedTerm(class={self.class_}, value={self.value}, start_pos={self.start_pos}, source={self.source})'

    def __eq__(self, other):
        if isinstance(other, ClassifiedTerm):
            return other.class_ == self.class_ and other.value == self.value

        return False

    def __hash__(self):
        return hash((self.class_, self.value))


class GroupedTerm:
    def __init__(self, class_name: str, terms: List[ClassifiedTerm]):
        if not terms:
            raise ValueError('Terms must be non-empty.')

        self.items = []
        self.class_ = class_name

        values = set()
        for term in terms:
            if ' ' in term.value:
                value = term.value
            else:
                value = ru_morph.parse(term.value)[0].normal_form

            if value in values:
                continue
            values.add(value)

            if term.value.lower() == value:
                value = term.value
            elif term.value.istitle():
                value = value[:1].upper() + value[1:]

            term.value = value
            self.items.append(term)

    def size(self) -> int:
        return len(self.items)

    def as_single(self) -> ClassifiedTerm:
        return self.items[0]

    def __repr__(self):
        return f'GroupedTerm(class={self.class_}, terms={self.items})'

    def __eq__(self, other):
        if isinstance(other, GroupedTerm):
            return other.class_ == self.class_ and other.items == self.items

        return False

    def __hash__(self):
        return hash((self.class_, self.items))


class Relation:
    def __init__(self, term1: ClassifiedTerm, predicate: str, term2: ClassifiedTerm):
        self.term1 = term1
        self.term2 = term2
        self.predicate = predicate
        self.id = f'{self.term1.class_}_{self.predicate}_{self.term2.class_}'

        if not predicate:
            raise ValueError('Got empty predicate.')

    def __repr__(self):
        return f'Relation(term1={self.term1}, predicate={self.predicate}, term2={self.term2})'

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False

        return self.term1 == other.term1 and self.term2 == other.term2 and self.predicate == other.predicate

    def __hash__(self):
        return hash((self.term1, self.predicate, self.term2))

    def as_str(self):
        return f'({self.term1.value}) {self.predicate} ({self.term2.value})'

    def to_json(self):
        return {
            'term1': self.term1.to_json(),
            'predicate': self.predicate,
            'term2': self.term2.to_json()
        }

    def inverse(self):
        return Relation(term1=self.term2, predicate=self.predicate, term2=self.term1)

    @staticmethod
    def from_json(rel_json: Dict[str, Any], text: str = ''):
        term1 = ClassifiedTerm.from_json(rel_json['term1'], text)
        term2 = ClassifiedTerm.from_json(rel_json['term2'], text)

        return Relation(
            term1=term1,
            predicate=rel_json['predicate'],
            term2=term2
        )


class Sentence:
    def __init__(self,
                 sent_id: int,
                 text: str,
                 terms: List[ClassifiedTerm],
                 relations: List[Relation]):
        self.id = sent_id
        self.text = text
        self.terms = terms
        self.relations = relations

    def find_relation_by_id(self, relation_id: str) -> Optional[Relation]:
        for r in self.relations:
            if f'{r.term1.class_}_{r.predicate}_{r.term2.class_}' == relation_id:
                return r
        return None

    def find_term_by_class(self, class_id: str) -> Optional[Term]:
        for term in self.terms:
            if term.class_ == class_id:
                return term

        return None

    def find_terms_by_class(self, class_id: str) -> List[Term]:
        return [t for t in self.terms if t.class_ == class_id]

    def to_json(self):
        return {
            'id': self.id,
            'text': self.text,
            'terms': [t.to_json() for t in self.terms],
            'relations': [r.to_json() for r in self.relations]
        }

    @staticmethod
    def from_json(sent_json: Dict[str, Any]):
        text = sent_json['text']
        id = sent_json['id']
        terms = [ClassifiedTerm.from_json(t, text) for t in sent_json['terms']]
        relations = [Relation.from_json(r, text) for r in sent_json['relations']]
        return Sentence(id, text, terms, relations)

    def __repr__(self):
        return f'Sentence(id={self.id}, text="{self.text}", terms={self.terms}, relations={self.relations})'

    def __hash__(self):
        return hash((self.text, self.terms, self.relations))


def read_sentences(file_name: str) -> List[Sentence]:
    with open(file_name, 'r', encoding='utf-8') as f:
        sentences_json = json.load(f)

        return [Sentence.from_json(s) for s in sentences_json['sentences']]
