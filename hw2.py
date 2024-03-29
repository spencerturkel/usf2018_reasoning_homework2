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
                      Tuple[str, Union[str, Predicate]*]
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
            result = (token,) + tuple(args)

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


def symbols_of(predicate):
    """
    :param predicate: the predicate to scan
    :return: the set of predicates, the set of functions, and the set of object
    """
    predicates = set()
    functions = set()
    objects = set()

    def _function(pred):
        if pred == 'CONTR':
            return
        tag = pred[0]
        if tag in ['FORALL', 'EXISTS']:
            objects.add(pred[1])
            _predicate(pred[2])
            return
        if tag in ['AND', 'OR', 'IMPLIES']:
            _predicate(pred[1])
            _predicate(pred[2])
            return
        if tag == 'NOT':
            _predicate(pred[1])
            return
        if pred not in predicates:
            functions.add(tag)
        for arg in pred[1:]:
            if isinstance(arg, str):
                objects.add(arg)
                continue
            _function(arg)

    def _predicate(pred):
        if pred == 'CONTR':
            return
        tag = pred[0]
        if tag in ['FORALL', 'EXISTS']:
            objects.add(pred[1])
            _predicate(pred[2])
            return
        if tag in ['AND', 'OR', 'IMPLIES']:
            _predicate(pred[1])
            _predicate(pred[2])
            return
        if tag == 'NOT':
            _predicate(pred[1])
            return
        predicates.add(tag)
        for arg in pred[1:]:
            if isinstance(arg, str):
                objects.add(arg)
                continue
            _function(arg)

    _predicate(predicate)
    return predicates, functions, objects


@unique
class SubProofKind(Enum):
    conditional = 10
    universal = 20
    existential = 30
    arbitrary = 40


def instantiate(obj, quantifier_predicate, fresh):
    def substitute(obj, var, pred):
        if pred == 'CONTR':
            return pred
        tag, *data = pred
        if tag in {'FORALL', 'EXISTS'}:
            quantified_variable, quantified_predicate = data
            if var == quantified_variable:
                return pred
            if obj == quantified_variable:
                if obj not in symbols_of(quantified_predicate)[2]:
                    return pred
                fresh_variable = fresh()
                quantified_predicate = substitute(fresh_variable, quantified_variable, quantified_predicate)
                quantified_variable = fresh_variable
            return tag, quantified_variable, substitute(obj, var, quantified_predicate)
        if tag in {'AND', 'OR', 'IMPLIES'}:
            first_arg, second_arg = data
            return tag, substitute(obj, var, first_arg), substitute(obj, var, second_arg)
        if tag == 'NOT':
            arg, = data
            return tag, substitute(obj, var, arg)

        result = [tag]
        for arg in data:
            if isinstance(arg, str):
                result.append(obj if arg == var else arg)
            else:
                result.append(substitute(obj, var, arg))

        return tuple(result)

    if isinstance(quantifier_predicate, str) or len(quantifier_predicate) != 3:
        raise InvalidProof
    quantifier, variable, predicate = quantifier_predicate
    if quantifier not in {'FORALL', 'EXISTS'}:
        raise InvalidProof
    return substitute(obj, variable, predicate)


def validate_proof(proof, facts_by_line, seen_predicates, seen_functions, seen_objects):
    proof_length = len(proof)

    if proof_length == 4:
        if isinstance(proof[1], str) and not isinstance(proof[2], list):  # existential constant
            index, variable, predicate, cited_index = proof
            seen_symbols = seen_predicates | seen_functions | seen_objects

            if index in facts_by_line \
                    or cited_index not in facts_by_line \
                    or variable in seen_symbols \
                    or variable == 'CONTR':
                raise InvalidProof

            instantiation = instantiate(variable, facts_by_line[cited_index], lambda: None)
            if instantiation != predicate:
                raise InvalidProof

            facts_by_line = facts_by_line.copy()
            facts_by_line[index] = predicate

            pred_syms, fun_syms, obj_syms = symbols_of(instantiation)
            seen_predicates = seen_predicates | pred_syms
            seen_functions = seen_functions | fun_syms
            seen_objects = seen_objects | obj_syms | {variable}

            return facts_by_line, seen_predicates, seen_functions, seen_objects

        else:  # proof line
            index, predicate, cited_indices, rule = proof

            if index in facts_by_line:
                raise InvalidProof

            try:
                citations = [facts_by_line[i] for i in cited_indices]
            except KeyError:
                raise InvalidProof

            if rule == 'CI':
                if len(predicate) != 3 or not (0 < len(cited_indices) < 3):
                    raise InvalidProof

                tag, first_arg, second_arg = predicate

                if tag != 'AND' or first_arg not in citations or second_arg not in citations:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate

                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'CE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [(tag, *disjuncts)] = citations

                if tag != 'AND' or predicate not in disjuncts:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate

                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'DI':
                if len(predicate) != 3 or len(cited_indices) != 1:
                    raise InvalidProof

                tag, first_arg, second_arg = predicate

                if tag != 'OR':
                    raise InvalidProof

                [cited_proof] = citations

                if cited_proof != first_arg and cited_proof != second_arg:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate

                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'DE':
                disjuncts = None
                antecedents = set()
                consequents = []

                for cited_proof in citations:
                    if len(cited_proof) != 3:
                        raise InvalidProof
                    tag, *rest = cited_proof

                    if tag == SubProofKind.conditional:
                        sub_antecedents, sub_consequents = rest
                        if len(sub_antecedents) != 1:
                            raise InvalidProof
                        [ant] = sub_antecedents
                        antecedents.add(ant)
                        consequents.append(sub_consequents)
                        continue

                    if tag != 'OR':
                        raise InvalidProof

                    disjuncts = set(rest)

                if not disjuncts \
                        or disjuncts != antecedents \
                        or len(cited_indices) != len(antecedents) + 1:
                    raise InvalidProof

                for consequent_set in consequents:
                    if predicate not in consequent_set:
                        raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate

                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'II':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [sub_proof] = citations

                if len(sub_proof) != 3:
                    raise InvalidProof

                kind, conditions, sub_consequents = sub_proof

                if kind != SubProofKind.conditional \
                        or len(conditions) != 1:
                    raise InvalidProof

                [condition] = conditions
                tag, antecedent, consequent = predicate

                if tag != 'IMPLIES' \
                        or condition != antecedent \
                        or consequent not in sub_consequents:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate

                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'IE':
                if len(cited_indices) != 2:
                    raise InvalidProof

                antecedent = None
                consequent = None
                antecedent_proof = None

                for cited_proof in citations:
                    if cited_proof[0] == 'IMPLIES':
                        _, antecedent, consequent = cited_proof
                        if predicate != consequent:
                            raise InvalidProof
                        continue

                    antecedent_proof = cited_proof

                if antecedent_proof is None \
                        or antecedent is None \
                        or consequent is None \
                        or antecedent_proof != antecedent:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = consequent
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'NI':
                if len(predicate) != 2 or len(cited_indices) != 1:
                    raise InvalidProof

                tag, negated_predicate = predicate

                if tag != 'NOT':
                    raise InvalidProof

                [cited_proof] = citations

                if len(cited_proof) != 3:
                    raise InvalidProof

                kind, conditions, sub_facts = cited_proof

                if kind != SubProofKind.conditional \
                        or len(conditions) != 1 \
                        or 'CONTR' not in sub_facts:
                    raise InvalidProof

                [cond] = conditions
                if cond != negated_predicate:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'NE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [cited_proof] = citations

                if len(cited_proof) != 2 \
                        or cited_proof[0] != 'NOT':
                    raise InvalidProof

                inner_negation = cited_proof[1]

                if len(inner_negation) != 2 \
                        or inner_negation[0] != 'NOT' \
                        or inner_negation[1] != predicate:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'XI':
                if len(cited_indices) != 2 or predicate != 'CONTR':
                    raise InvalidProof

                [first_citation, second_citation] = citations

                if first_citation != ('NOT', second_citation) \
                        and second_citation != ('NOT', first_citation):
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'XE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [contradiction] = citations

                if contradiction != 'CONTR':
                    raise InvalidProof

                pred_syms, fun_syms, obj_syms = symbols_of(predicate)

                if not (pred_syms.issubset(seen_predicates) and
                        fun_syms.issubset(seen_functions) and
                        obj_syms.issubset(seen_objects)):
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'AI':
                if len(cited_indices) != 1 or len(predicate) != 3:
                    raise InvalidProof

                tag, var, _ = predicate

                if tag != 'FORALL' or var in seen_predicates | seen_functions | seen_objects:
                    raise InvalidProof

                [cited_proof] = citations

                if len(cited_proof) != 3:
                    raise InvalidProof

                kind, uconst, props = cited_proof

                if kind != SubProofKind.universal:
                    raise InvalidProof

                if instantiate(uconst, predicate, lambda: None) not in props:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects | {var}

            if rule == 'AE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [cited_proof] = citations

                if len(cited_proof) != 3:
                    raise InvalidProof

                tag, _, _ = cited_proof

                if tag != 'FORALL':
                    raise InvalidProof

                for obj in seen_objects:
                    if instantiate(obj, cited_proof, lambda: None) == predicate:
                        facts = facts_by_line.copy()
                        facts[index] = predicate
                        return facts, seen_predicates, seen_functions, seen_objects

                raise InvalidProof

            if rule == 'EI':
                if len(cited_indices) != 1 or len(predicate) != 3:
                    raise InvalidProof

                tag, var, _ = predicate

                if tag != 'EXISTS' or var in seen_predicates | seen_functions | seen_objects:
                    raise InvalidProof

                [cited_proof] = citations

                _, _, cited_proof_objects = symbols_of(cited_proof)

                for obj in cited_proof_objects:
                    if instantiate(obj, predicate, lambda: None) == cited_proof:
                        facts = facts_by_line.copy()
                        facts[index] = predicate
                        return facts, seen_predicates, seen_functions, seen_objects | {var}

                raise InvalidProof

            if rule == 'EE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [cited_proof] = citations

                kind, sub_facts = cited_proof

                if kind != SubProofKind.existential or predicate not in sub_facts:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'S':
                if cited_indices or predicate in facts_by_line.values():
                    raise InvalidProof

                pred_syms, fun_syms, obj_syms = symbols_of(predicate)

                if 'CONTR' in pred_syms | fun_syms \
                        or pred_syms & (seen_functions | seen_objects | fun_syms | obj_syms) \
                        or fun_syms & (seen_predicates | seen_objects | pred_syms | obj_syms) \
                        or obj_syms & (seen_predicates | seen_functions | pred_syms | fun_syms):
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                seen_predicates = seen_predicates | pred_syms
                seen_functions = seen_functions | fun_syms
                seen_objects = seen_objects | obj_syms
                return facts, seen_predicates, seen_functions, seen_objects

            if rule == 'RE':
                if len(cited_indices) != 1:
                    raise InvalidProof

                [cited_proof] = citations

                if cited_proof != predicate:
                    raise InvalidProof

                facts = facts_by_line.copy()
                facts[index] = predicate
                return facts, seen_predicates, seen_functions, seen_objects

    if proof_length == 2:
        if isinstance(proof[1], str):  # universal constant
            index, variable = proof
            if index in facts_by_line \
                    or variable in seen_predicates | seen_functions | seen_objects \
                    or variable == 'CONTR':
                raise InvalidProof

            return facts_by_line, seen_predicates, seen_functions, seen_objects | {variable}

        else:  # sub-proof
            index, sub_proof_list = proof

            if index in facts_by_line:
                raise InvalidProof

            outer_facts = facts_by_line.copy()
            kind = SubProofKind.arbitrary
            const = None
            antecedents = set()

            for sub_proof in sub_proof_list:
                if len(sub_proof) == 4:
                    last = sub_proof[3]
                    if last == 'S':
                        antecedents.add(sub_proof[1])
                        kind = SubProofKind.conditional
                    elif isinstance(last, int):
                        kind = SubProofKind.existential
                        const = sub_proof[1]
                else:
                    last = sub_proof[1]
                    if isinstance(last, str):
                        kind = SubProofKind.universal
                        const = last

                facts_by_line, seen_predicates, seen_functions, seen_objects = validate_proof(
                    sub_proof, facts_by_line, seen_predicates,
                    seen_functions, seen_objects)

            sub_facts = set()
            for k, v in facts_by_line.items():
                if k in outer_facts:
                    continue
                tag = v[0]
                if tag == SubProofKind.arbitrary:
                    sub_facts |= v[1]
                    continue
                if tag in [SubProofKind.conditional, SubProofKind.existential, SubProofKind.universal]:
                    continue
                _, _, obj_syms = symbols_of(v)
                if kind == SubProofKind.existential and const in obj_syms:
                    continue
                sub_facts.add(v)

            if kind == SubProofKind.arbitrary:
                outer_facts[index] = (SubProofKind.arbitrary, sub_facts)
            if kind == SubProofKind.conditional:
                outer_facts[index] = (SubProofKind.conditional, antecedents, sub_facts)
            if kind == SubProofKind.universal:
                outer_facts[index] = (SubProofKind.universal, const, sub_facts)
            if kind == SubProofKind.existential:
                outer_facts[index] = (SubProofKind.existential, sub_facts)
            return outer_facts, seen_predicates, seen_functions, seen_objects

    raise InvalidProof


def validate(top_level_proof):
    """
    Validates a top-level proof.

    :param top_level_proof: a parsed Proof.
    :except InvalidProof: when the proof is valid
    """
    if len(top_level_proof) != 2 or isinstance(top_level_proof[1], str):
        raise InvalidProof
    top_level_line, _ = top_level_proof

    facts_by_line, _, _, _ = validate_proof(top_level_proof, dict(), set(), set(), set())

    if top_level_line not in facts_by_line:
        raise InvalidProof


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
