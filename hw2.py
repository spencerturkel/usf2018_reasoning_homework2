"""
Created by Spencer Turkel on 09/27/2018.
"""
import random
import re
from enum import Enum, unique
from string import whitespace


class InputSyntaxError(Exception):
    pass


class Lexer:
    """
    An iterable class for lexing.
    You can also peek at the next character (excluding whitespace).
    """

    _word_regex = re.compile('[A-Za-z0-9]+')

    def __init__(self, text):
        self.length = len(text)
        self.index = 0
        self.text = text

    def __iter__(self):
        return self

    def __next__(self):
        next_token = self._next_lexeme()
        if not next_token:
            raise StopIteration
        self.index += len(next_token)
        return int(next_token) if next_token.isdigit() else next_token

    def peek(self):
        """
        Peeks at the next lexeme in the text.
        :return: The next lexeme in the text, or None if there are none.
        """
        next_token = self._next_lexeme()
        if not next_token:
            return None
        return int(next_token) if next_token.isdigit() else next_token

    def _next_lexeme(self):
        """
        Peeks at the next lexeme in the text.
        :return: The next lexeme in the text, or None if there are none.
        """
        next_char = self._next_char()
        if not next_char:
            return None
        if next_char in {'(', ')', '[', ']', ','}:
            return next_char
        match = self._word_regex.match(self.text, self.index)
        return match.group(0) if match else None

    def _next_char(self):
        while True:
            if self.index == self.length:
                return None
            ch = self.text[self.index]
            if ch not in whitespace:
                return ch
            self.index += 1


class ParseError(Exception):
    pass


def parse(lexer):
    """
    This is a recursive descent parser for the input lexemes, returning a structured Proof.

    :param lexer: an iterator of lexemes allowing a .peek() operation for lookahead
    :return: a Proof.
    Proof = Union[Tuple[int, List[Proof]],
                  Tuple[int, Predicate, List[int], str],
                  Tuple[int, str],
                  Tuple[int, str, Predicate, int]]
    Predicate = Union[Tuple['FORALL', str, Predicate],
                      Tuple['EXISTS', str, Predicate],
                      Tuple['AND', Predicate, Predicate],
                      Tuple['OR', Predicate, Predicate],
                      Tuple['IMPLIES', Predicate, Predicate],
                      Tuple['NOT', Predicate],
                      'CONTR',
                      Tuple[str, List[Union[str, Predicate]]],
    :except ParseError: when the proof cannot be parsed.
    """

    def _expect_next(token):
        if next(lexer) != token:
            raise ParseError

    def _proof():
        _expect_next('(')
        line = _line()
        _expect_next(')')
        return line

    def _line():
        token = next(lexer)

        if token == 'SUBP':
            num = _index()
            proofs = []
            while lexer.peek() != ')':
                proofs.append(_proof())
            return num, proofs

        if not isinstance(token, int):
            raise ParseError
        index = token

        if next(lexer) != '(':
            raise ParseError

        if lexer.peek() == 'UCONST':
            next(lexer)
            symbol = next(lexer)
            for token in ') ( [ ] UCONST )'.split(' '):
                _expect_next(token)
            return index, symbol

        if lexer.peek() == 'ECONST':
            next(lexer)
            symbol = next(lexer)
            predicate = _predicate()
            for token in (')', '(', '['):
                _expect_next(token)
            cited_index = _index()
            for token in (']', 'ECONST', ')'):
                _expect_next(token)
            return index, symbol, predicate, cited_index

        predicate = _predicate_after_open_paren()
        _expect_next('(')
        _expect_next('[')
        cited_indices = _indices()
        _expect_next(']')
        rule = _rule()
        _expect_next(')')
        return index, predicate, cited_indices, rule

    def _index():
        token = next(lexer)
        if not isinstance(token, int):
            raise ParseError
        return token

    def _predicate():
        _expect_next('(')
        return _predicate_after_open_paren()

    def _predicate_after_open_paren():
        token = _symbol()

        if token in {'FORALL', 'EXISTS'}:
            symbol = _symbol()
            predicate = _predicate()
            result = token, symbol, predicate
        elif token in {'AND', 'OR', 'IMPLIES'}:
            result = token, _predicate(), _predicate()
        elif token == 'NOT':
            result = token, _predicate()
        elif token == 'CONTR':
            result = token
        else:
            args = []
            while lexer.peek() != ')':
                args.append(_predicate() if lexer.peek() == '(' else _symbol())
            result = token, args

        _expect_next(')')
        return result

    def _indices():
        indices = []
        if lexer.peek() != ']':
            indices.append(_index())
            while lexer.peek() == ',':
                next(lexer)
                indices.append(_index())
        return indices

    def _rule():
        rule = next(lexer)
        if rule not in {'S', 'RE'} | {x + y for x in 'CDINXAE' for y in 'IE'}:
            raise ParseError
        return rule

    def _symbol():
        token = next(lexer)
        if not isinstance(token, str):
            raise ParseError
        return token

    return _proof()


class InvalidProof(Exception):
    pass


def validate(top_level_proof):
    """
    Validates a top-level proof.

    :param top_level_proof: a Proof.
    Proof = Union[Tuple[int, List[Proof]],
                  Tuple[int, Predicate, List[int], str],
                  Tuple[int, str],
                  Tuple[int, str, Predicate, int]]
    Predicate = Union[Tuple['FORALL', str, Predicate],
                      Tuple['EXISTS', str, Predicate],
                      Tuple['AND', Predicate, Predicate],
                      Tuple['OR', Predicate, Predicate],
                      Tuple['IMPLIES', Predicate, Predicate],
                      Tuple['NOT', Predicate],
                      'CONTR',
                      Tuple[str, List[Union[str, Predicate]]],
    :except InvalidProof: when the proof is valid
    """
    if len(top_level_proof) != 2 or isinstance(top_level_proof[1], str):
        raise InvalidProof
    top_level_line, _ = top_level_proof

    def _validate_proof(proof, facts_by_line, predicates, functions_and_objects):
        return facts_by_line, predicates, functions_and_objects

    _validate_proof(top_level_proof, dict(), set(), set())


# noinspection PyPep8Naming
def verifyProof(P):
    """
    :param P: A string which is an S-expression of a well-formed Fitch-style
    proof.
    :return: Returns either:
        “I” – If P was well-formed, but not a valid proof,
        “V” – If P was well-formed, and a valid proof.
    """
    try:
        validate(parse(Lexer(P)))
        return 'V'
    except InvalidProof:
        return 'I'
