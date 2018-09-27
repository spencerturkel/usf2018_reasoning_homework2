"""
Created by Spencer Turkel on 09/27/2018.
"""

from enum import Enum, unique
from re import compile as compile_regex


# first, defining some token enums for the lexer
from string import whitespace


@unique
class QuantifiedConstant(Enum):
    universal_constant = 1
    existential_constant = 2


@unique
class Op(Enum):
    universal = 1
    existence = 2
    contradiction = 3
    conjunction = 4
    disjunction = 5
    implication = 6
    negation = 7


@unique
class InferenceRule(Enum):
    supposition = 1
    conjunction_introduction = 2
    conjunction_elimination = 3
    disjunction_introduction = 4
    disjunction_elimination = 5
    implication_introduction = 6
    implication_elimination = 7
    negation_introduction = 8
    negation_elimination = 9
    universal_introduction = 10
    universal_elimination = 11
    existential_introduction = 12
    existential_elimination = 13
    contradiction_introduction = 14
    contradiction_elimination = 15


@unique
class CommonToken(Enum):
    left_parenthesis = 1
    right_parenthesis = 2
    left_bracket = 3
    right_bracket = 4
    comma = 5
    sub_proof = 6


# regular expressions compiled for lexing
index_regex = compile_regex(r'\d+')
object_regex = compile_regex(r'[a-z0-9]+')
predicate_regex = compile_regex(r'[A-Z][a-z0-9]+')


class InputSyntaxError(Exception):
    pass


class Lexer:
    """
    An iterable class for lexing.
    You can also peek at the next character (excluding whitespace).
    """

    def __init__(self, text):
        self.length = len(text)
        self.index = 0
        self.text = text

    def __iter__(self):
        return self

    def peek(self):
        """
        Peeks at the next non-whitespace character in the text.
        :return: The next non-whitespace character in the text.
        :except InputSyntaxError: if there are no more non-whitespace characters
        """
        while True:
            ch = self.text[self.index]
            if ch not in whitespace:
                return ch
            self.index += 1
            if self.index == self.length:
                raise InputSyntaxError

    def _next_word_is(self, word):
        return self.text.startswith(word, self.index)

    def __next__(self):
        if self.index == self.length:
            raise StopIteration
        next_char = self.peek()
        if next_char == '(':
            self.index += len('(')
            return CommonToken.left_parenthesis
        if next_char == ')':
            self.index += len(')')
            return CommonToken.right_parenthesis
        if next_char == '[':
            self.index += len('[')
            return CommonToken.left_bracket
        if next_char == ']':
            self.index += len(']')
            return CommonToken.right_bracket
        if next_char == ',':
            self.index += len(',')
            return CommonToken.comma
        if self._next_word_is('SUBP'):
            self.index += len('SUBP')
            return CommonToken.sub_proof
        if self._next_word_is('UCONST'):
            self.index += len('UCONST')
            return QuantifiedConstant.universal_constant
        if self._next_word_is('ECONST'):
            self.index += len('ECONST')
            return QuantifiedConstant.existential_constant
        if self._next_word_is('FORALL'):
            self.index += len('FORALL')
            return Op.universal
        if self._next_word_is('EXISTS'):
            self.index += len('EXISTS')
            return Op.existence
        if self._next_word_is('CONTR'):
            self.index += len('CONTR')
            return Op.contradiction
        if self._next_word_is('AND'):
            self.index += len('AND')
            return Op.conjunction
        if self._next_word_is('OR'):
            self.index += len('OR')
            return Op.disjunction
        if self._next_word_is('IMPLIES'):
            self.index += len('IMPLIES')
            return Op.implication
        if self._next_word_is('NOT'):
            self.index += len('NOT')
            return Op.negation
        if self._next_word_is('S'):
            self.index += len('S')
            return InferenceRule.supposition
        if self._next_word_is('CI'):
            self.index += len('CI')
            return InferenceRule.conjunction_introduction
        if self._next_word_is('CE'):
            self.index += len('CE')
            return InferenceRule.conjunction_elimination
        if self._next_word_is('DI'):
            self.index += len('DI')
            return InferenceRule.disjunction_introduction
        if self._next_word_is('DE'):
            self.index += len('DE')
            return InferenceRule.disjunction_elimination
        if self._next_word_is('II'):
            self.index += len('II')
            return InferenceRule.implication_introduction
        if self._next_word_is('IE'):
            self.index += len('IE')
            return InferenceRule.implication_elimination
        if self._next_word_is('NI'):
            self.index += len('NI')
            return InferenceRule.negation_introduction
        if self._next_word_is('NE'):
            self.index += len('NE')
            return InferenceRule.negation_elimination
        if self._next_word_is('AI'):
            self.index += len('AI')
            return InferenceRule.universal_introduction
        if self._next_word_is('AE'):
            self.index += len('AE')
            return InferenceRule.universal_elimination
        if self._next_word_is('EI'):
            self.index += len('EI')
            return InferenceRule.existential_introduction
        if self._next_word_is('EE'):
            self.index += len('EE')
            return InferenceRule.existential_elimination
        if self._next_word_is('XI'):
            self.index += len('XI')
            return InferenceRule.contradiction_introduction
        if self._next_word_is('XE'):
            self.index += len('XE')
            return InferenceRule.contradiction_elimination
        raise InputSyntaxError


def lex(proof):
    """
    This is a lexer for the input proofs.

    :param proof: a Fitch-style proof represented as an s-expression. See assignment or README.
    :return:
        a generator of lexemes,
        using previously defined tokens instead of strings where appropriate.
    """
    index = 0
    while True:
        item = proof[index]
        if item in whitespace:
            continue
        elif item == '(':
            yield CommonToken.left_parenthesis
        elif item == ')':
            yield CommonToken.right_parenthesis
        elif item == '[':
            yield CommonToken.left_bracket
        elif item == ']':
            yield CommonToken.right_bracket
        elif item == ',':
            yield CommonToken.right_bracket

def parse(lexemes):
    """
    This is a recursive descent parser for the input proofs.

    :param lexemes: a Fitch-style proof represented as a generator of lexemes.
    :return:
        nested lists with identical structure to the input proof,
        using previously defined tokens instead of strings where appropriate.
    """


# noinspection PyPep8Naming
def verifyProof(P):
    """
    :param P: A string which is an S-expression of a well-formed Fitch-style
    proof.
    :return: Returns either:
        “I” – If P was well-formed, but not a valid proof,
        “V” – If P was well-formed, and a valid proof.
    """
    return 'I'
