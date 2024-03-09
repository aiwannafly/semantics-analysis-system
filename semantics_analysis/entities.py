from typing import Dict, List, Optional


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
            return other.value == self.value and other.start_pos == self.start_pos and other.end_pos == self.end_pos

        return False

    def __hash__(self):
        return hash((self.value, self.start_pos, self.end_pos, self.text))


class ClassifiedTerm(Term):

    def __init__(
            self,
            term_class: str,
            value: str,
            end_pos: int,
            text: str
    ):
        super().__init__(value, end_pos, text)
        self.class_ = term_class

    @classmethod
    def from_term(cls, term_class: str, term: Term):
        return cls(term_class, term.value, term.end_pos, term.text)

    def __repr__(self):
        return f'ClassifiedTerm(class={self.class_}, value={self.value}, start_pos={self.start_pos}, end_pos={self.end_pos})'

    def __eq__(self, other):
        if isinstance(other, ClassifiedTerm):
            return (other.class_ == self.class_ and other.value == self.value and other.start_pos == self.start_pos
                    and other.end_pos == self.end_pos)

        return False

    def __hash__(self):
        return hash((self.class_, self.value, self.start_pos, self.end_pos, self.text))


class Relation:
    def __init__(self, term1: ClassifiedTerm, predicate: str, term2: ClassifiedTerm):
        self.term1 = term1
        self.term2 = term2
        self.predicate = predicate

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

    def find_relation_by_id(self, relation_id: str) -> Optional[Relation]:
        for r in self.relations:
            if f'{r.term1.class_}_{r.predicate}_{r.term2.class_}' == relation_id:
                return r
        return None

    def find_term_by_class_id(self, class_id: str) -> Optional[str]:
        if class_id not in self.terms:
            return None

        return self.terms[class_id][0]

    def __repr__(self):
        return f'Sentence(id={self.id}, text="{self.text}", terms={self.terms}, relations={self.relations})'
