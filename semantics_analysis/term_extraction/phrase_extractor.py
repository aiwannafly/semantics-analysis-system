from typing import List, Optional

import spacy
from alphabet_detector import AlphabetDetector

# POS
ADJ = 'ADJ'
NOUN = 'NOUN'
PROPN = 'PROPN'
ADP = 'ADP'
CCONJ = 'CCONJ'
NUMBER = 'NUM'
SYMBOL = 'SYM'
UNKNOWN = 'X'

# Number
PLURAL = 'Plur'
SINGLE = 'Sing'

# Case
NOM = 'Nom'
ACC = 'Acc'
DAT = 'Dat'
INS = 'Ins'
GEN = 'Gen'


class Morph:
    pos: str
    case: Optional[str]
    number: Optional[str]
    value: Optional[str]

    def __init__(self, pos: str, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        self.pos = pos
        self.case = case
        self.number = number
        self.value = value

    def __repr__(self):
        return f'Morph(pos={self.pos}, case={self.case}, numer={self.number}, value={self.value})'


class Noun(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(NOUN, case, number, value)


class Adjective(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(ADJ, case, number, value)


class Preposition(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(ADP, case, number, value)


class Conjunction(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(CCONJ, case, number, value)


class Number(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(NUMBER, case, number, value)


class Symbol(Morph):
    def __init__(self, case: Optional[str] = None, number: Optional[str] = None, value: Optional[str] = None):
        super().__init__(SYMBOL, case, number, value)


class Rule:
    morphs: List[Morph]

    def __init__(self, *morphs: Morph):
        self.morphs = list(morphs)

    def __len__(self):
        return len(self.morphs)


rules = [
    Rule(Adjective(), Adjective(), Noun()),
    Rule(Adjective(), Noun()),
    Rule(Adjective(), Noun(), Noun(case=GEN)),

    Rule(Adjective(number=PLURAL), Noun(), Preposition(value='с'), Noun(case=INS)),
    Rule(Adjective(number=PLURAL), Noun(), Conjunction(value='и'), Noun()),

    Rule(Noun(), Adjective()),
    Rule(Noun(), Adjective(case=GEN), Noun(case=GEN)),
    Rule(Noun(), Preposition(), Adjective(), Noun()),
    Rule(Noun(), Preposition(), Noun()),
    Rule(Noun(), Preposition(), Noun(), Noun()),

    Rule(Noun(), Noun(), Adjective()),
    Rule(Noun(), Noun(case=DAT)),
    Rule(Noun(), Noun(case=NOM)),
    Rule(Noun(), Noun(case=GEN)),

    Rule(Noun(), Noun(case=GEN), Preposition(value='с'), Noun(case=INS)),
    Rule(Noun(), Noun(case=GEN), Conjunction(value='и'), Noun(case=GEN)),

    Rule(Noun(), Noun(case=GEN), Noun(case=GEN)),
    Rule(Noun(), Noun(case=GEN), Noun(case=GEN), Noun(case=GEN)),

    Rule(Number(), Symbol()),
    Rule(Number()),

    Rule(Noun())
]


class PhraseExtractor:

    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.nlp.disable_pipes(["parser", "attribute_ruler", "lemmatizer"])

        self.rules = sorted(rules, key=len, reverse=True)
        self.ad = AlphabetDetector()

    def __call__(self, text: str) -> List[str]:
        found_phrases = []

        doc = self.nlp(text)

        found_morphs = self._parse_text(doc)

        morphs_len = len(found_morphs)

        already_taken = [False] * morphs_len

        for rule in self.rules:
            rule_len = len(rule.morphs)
            rule_morphs = rule.morphs

            for i in range(0, morphs_len - rule_len + 1):
                text_morphs = found_morphs[i:i + rule_len]

                if None in text_morphs:
                    continue

                part_already_taken = False

                for j in range(i, i + rule_len):
                    if already_taken[j]:
                        part_already_taken = True
                        break

                if part_already_taken:
                    continue

                matches = True
                for rule_morph, text_morph in zip(rule_morphs, text_morphs):
                    if rule_morph.pos != text_morph.pos:
                        matches = False
                        break

                    if rule_morph.case is not None and rule_morph.case != text_morph.case:
                        matches = False
                        break

                    if rule_morph.number is not None and rule_morph.number != text_morph.number:
                        matches = False
                        break

                    if rule_morph.value is not None and rule_morph.value != text_morph.value:
                        matches = False
                        break

                if matches:
                    for j in range(i, i + rule_len):
                        already_taken[j] = True

                    if rule_len == 2 and rule_morphs[1].pos == SYMBOL:
                        num, symbol = text_morphs[0].value, text_morphs[1].value

                        if f'{num} {symbol}' not in text:
                            found_phrases.append(f'{num}{symbol}')
                        else:
                            found_phrases.append(f'{num} {symbol}')
                    else:
                        found_phrases.append(' '.join(t.value for t in text_morphs))

        return found_phrases + self._find_english_phrases(doc)

    def _find_english_phrases(self, doc) -> List[str]:
        english_phrases = []

        curr_phrase = []

        for token in doc:
            if token.pos_ == UNKNOWN:
                curr_phrase.append(token.text)
            elif curr_phrase:
                english_phrases.append(' '.join(curr_phrase))
                curr_phrase = []

        if curr_phrase:
            english_phrases.append(' '.join(curr_phrase))

        return english_phrases

    def _parse_text(self, doc) -> List[Optional[Morph]]:
        morphs = []

        for token in doc:
            pos = token.pos_
            cases = token.morph.get('Case')
            numbers = token.morph.get('Number')

            number = None if not numbers else numbers[0]

            case = None if not cases else cases[0]

            if pos == NOUN or pos == PROPN:
                morphs.append(Noun(case=case, number=number, value=token.text))
            elif pos == ADJ:
                morphs.append(Adjective(case=case, number=number, value=token.text))
            elif pos == CCONJ:
                morphs.append(Conjunction(value=token.text))
            elif pos == ADP:
                morphs.append(Preposition(value=token.text))
            elif pos == NUMBER:
                morphs.append(Number(value=token.text))
            elif pos == SYMBOL:
                morphs.append(Symbol(value=token.text))
            else:
                morphs.append(None)

        return morphs


def main():
    extractor = PhraseExtractor()

    text = 'Мы взяли enwik8 — аккуратно очищенные статьи Википедии на английском языке. Эти тексты мы прогнали через изучаемые модели, сохраняя все промежуточные активации (для каждого токена и с каждого слоя). Так мы получили «пространство эмбеддингов» или, другими словами, многомерное облако точек, с которым и стали дальше работать.'
    # text = 'Мы использовали Tomita parser, в конечном итоге F1-мера составила 67,3%.'

    phrases = extractor(text)

    print(phrases)


if __name__ == '__main__':
    main()
