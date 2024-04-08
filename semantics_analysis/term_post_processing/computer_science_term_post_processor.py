from typing import List

from semantics_analysis.entities import ClassifiedTerm
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor

OS_NAMES = ['macOS', 'Linux', 'ChromeOS', 'Android', 'IOS', 'Unix',
            'Ubuntu', 'Fedora', 'Solaris', 'Symbian', 'Debian', 'Zorin', 'FreeBSD', 'MSDOS', 'AMSDOS',
            'Arch', 'Arch Linux', 'Redhat', 'GNU', 'OpenSolaris',
            'Apple', 'watchOS', 'Windows', 'Microsoft Windows', 'Windows Vista', 'Windows XP', 'Windows 7',
            'Windows 8', 'Windows 10', 'Windows 11', 'Windows NT', 'Windows Server', 'Windows IoT',
            'линукс', 'виндовс', 'убунту', 'дебиан', 'андроид', 'макос']

PROGRAMMING_LANGUAGES = [
    'Fortran', 'ALGOL', 'Bash', 'Lisp', 'C', 'Ada', 'Assembler', 'Pascal', 'C++', 'Java', 'PHP', 'Perl', 'Python',
    'Python 2', 'Python 3', 'BASIC', 'Visual BASIC', 'Clojure', 'COBOL', 'Cython', 'Delphi', 'Erlang', 'F#', 'Scala',
    'Kotlin', 'Go', 'Java 8', 'Java 17', 'Java 21',
    'C#', 'Rust', 'flow9', 'Haskell', 'D', 'Dart', 'Javascript', 'Typescript', 'Groovy', 'SQL',
    'питон', 'си', 'си++', 'джава', 'C/C++', 'C\\C++'
]


class ComputerScienceTermPostProcessor(TermPostProcessor):
    environment_terms = set([t.lower() for t in OS_NAMES + PROGRAMMING_LANGUAGES])

    def __call__(self, terms: List[ClassifiedTerm]) -> List[ClassifiedTerm]:
        processed_terms = []

        # some environment terms actually have 'Library' class
        for term in terms:
            if term.class_ != 'Environment':
                processed_terms.append(term)
                continue

            if term.value.lower() in self.environment_terms:
                processed_terms.append(term)
            else:
                processed_terms.append(ClassifiedTerm(
                    value=term.value,
                    term_class='Library',
                    end_pos=term.end_pos,
                    text=term.text
                ))

        return processed_terms
