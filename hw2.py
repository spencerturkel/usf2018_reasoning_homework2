"""
Created by Spencer Turkel on 09/27/2018.
"""

from enum import Enum, unique
from re import compile as compile_regex


# first, defining some token enums for the lexer

@unique
class Op(Enum):
    universal = 1
    existence = 2
    universal_constant = 3
    existential_constant = 4
    contradiction = 5
    conjunction = 6
    disjunction = 7
    implication = 8
    negation = 9


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
    universal_constant = 16
    existential_constant = 17


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
