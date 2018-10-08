"""
Created by Spencer Turkel on 09/27/2018.
"""
import re
from enum import Enum, unique
from string import whitespace


@unique
class QuantifiedConstant(Enum):
    universal = 1
    existential = 2

    def __repr__(self):
        return 'QuantifiedConstant.{0}'.format(self.name)


@unique
class Op(Enum):
    universal = 3
    existence = 4
    contradiction = 5
    conjunction = 6
    disjunction = 7
    implication = 8
    negation = 9

    def __repr__(self):
        return 'Op.{0}'.format(self.name)


@unique
class InferenceRule(Enum):
    supposition = 10
    conjunction_introduction = 11
    conjunction_elimination = 12
    disjunction_introduction = 13
    disjunction_elimination = 14
    implication_introduction = 15
    implication_elimination = 16
    negation_introduction = 17
    negation_elimination = 18
    universal_introduction = 19
    universal_elimination = 20
    existential_introduction = 21
    existential_elimination = 22
    contradiction_introduction = 23
    contradiction_elimination = 24
    reiteration = 25

    def __repr__(self):
        return 'InferenceRule.{0}'.format(self.name)


@unique
class CommonToken(Enum):
    left_parenthesis = 26
    right_parenthesis = 27
    left_bracket = 28
    right_bracket = 29
    comma = 30
    sub_proof = 31

    def __repr__(self):
        return 'CommonToken.{0}'.format(self.name)


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
            for token in ') ( ['.split(' '):
                _expect_next(token)
            cited_index = _index()
            for token in '] ECONST )'.split(' '):
                _expect_next(token)
            return index, symbol, predicate, cited_index

        predicate = _predicate_after_open_paren()
        cited_indices = _indices()
        rule = _rule()
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
            return token, symbol, predicate

        if token in {'AND', 'OR', 'IMPLIES'}:
            return token, _predicate(), _predicate()

        if token == 'NOT':
            return token, _predicate()

        if token == 'CONTR':
            return token

        args = []
        while lexer.peek() != ')':
            args.append(_predicate() if lexer.peek() == '(' else _symbol())

        return token, args

    def _indices():
        _expect_next('(')
        _expect_next('[')
        indices = []
        if lexer.peek() != ']':
            indices.append(_index())
            while lexer.peek() == ',':
                next(lexer)
                indices.append(_index())
        _expect_next(']')
        return indices

    def _rule():
        rule = next(lexer)
        if rule not in {'RE'} | {x + y for x in 'CDINXAE' for y in 'IE'}:
            raise ParseError
        return rule

    def _symbol():
        token = next(lexer)
        if not isinstance(token, str):
            raise ParseError
        return token


class Parser:
    def __init__(self):
        self._scanner = None

    def parse(self, text):
        """
        This is a recursive descent parser for the input lexemes, returning a structured Proof.

        :param text: the input
        :return: an element of type Proof (defined below).
        :except ParseError: when the proof cannot be parsed.

        Result type::

            Proof = Union[Tuple[int, List[Proof]], Tuple[int, Expr, Justification]]
            Expr = Union[ str,
                          Tuple[Op.universal,  str,  Expr],
                          Tuple[Op.existence,  str,  Expr],
                          Tuple[QuantifiedConstant.universal,  str],
                          Tuple[QuantifiedConstant.existential,  str,  Expr],
                          Tuple[Op.conjunction,  Expr,  Expr],
                          Tuple[Op.disjunction,  Expr,  Expr],
                          Tuple[Op.implication,  Expr,  Expr],
                          Tuple[Op.negation,  Expr],
                          Op.contradiction,
                          Tuple[str, Expr*],
                        ]
            Justification = List[int] * InferenceRule

        >>> p = Parser()
        >>> p.parse('(10 (AND p (q x)) ([5] S))')
        (10, (Op.conjunction, 'p', ('q', 'x')), ([5], InferenceRule.supposition))
        >>> p.parse('(SUBP 5 (10 (IMPLIES p (q x)) ([] S)) (20 p ([] S)) (30 (q x) ([10, 20] IE)))')
        (5, [(10, (Op.implication, 'p', ('q', 'x')), ([], InferenceRule.supposition)), (20, 'p', ([], InferenceRule.supposition)), (30, ('q', 'x'), ([10, 20], InferenceRule.implication_elimination))])
        >>> p.parse('(10 (OR p q) ([] S))')
        (10, (Op.disjunction, 'p', 'q'), ([], InferenceRule.supposition))
        >>> p.parse('(10 (FORALL p (q r)) ([] S))')
        (10, (Op.universal, 'p', ('q', 'r')), ([], InferenceRule.supposition))
        >>> p.parse('(10 (EXISTS p (q r)) ([] S))')
        (10, (Op.existence, 'p', ('q', 'r')), ([], InferenceRule.supposition))
        >>> p.parse('(10 (UCONST p) ([] UCONST))')
        (10, (QuantifiedConstant.universal, 'p'), ([], QuantifiedConstant.universal))
        >>> p.parse('(10 (ECONST p (q p)) ([] ECONST))')
        (10, (QuantifiedConstant.existential, 'p', ('q', 'p')), ([], QuantifiedConstant.existential))
        >>> p.parse('(10 (NOT p) ([] S))')
        (10, (Op.negation, 'p'), ([], InferenceRule.supposition))
        >>> p.parse('(10 (p x y) ([] S))')
        (10, ('p', 'x', 'y'), ([], InferenceRule.supposition))
        >>> p.parse('(10 (p) ([] S))')
        (10, ('p',), ([], InferenceRule.supposition))
        >>> p.parse('(10 (CONTR) ([] S))')
        (10, Op.contradiction, ([], InferenceRule.supposition))
        """
        self._scanner = Lexer(text)
        return self._proof()

    def _proof(self):
        self._expect_next(CommonToken.left_parenthesis)
        line = self._line()
        self._expect_next(CommonToken.right_parenthesis)
        return line

    def _expect_next(self, token):
        if next(self._scanner) != token:
            raise ParseError

    def _line(self):
        token = self._next()
        if token == CommonToken.sub_proof:
            return self._next(), self._many_proofs()
        if isinstance(token, int):
            return token, self._expr(), self._justification()
        raise ParseError

    def _next(self):
        return next(self._scanner)

    def _many_proofs(self):
        many_proofs = []
        while self._peek() not in [CommonToken.right_parenthesis, None]:
            many_proofs.append(self._proof())
        return many_proofs

    def _peek(self):
        return self._scanner.peek()

    def _expr(self):
        token = self._next()
        if token == CommonToken.left_parenthesis:
            formula = self._formula()
            self._expect_next(CommonToken.right_parenthesis)
            return formula
        if isinstance(token, str):
            return token
        raise ParseError

    def _justification(self):
        self._expect_next(CommonToken.left_parenthesis)
        self._expect_next(CommonToken.left_bracket)
        indices = self._indices()
        rule = self._next()
        if not isinstance(rule, InferenceRule) and not isinstance(rule, QuantifiedConstant):
            raise ParseError
        self._expect_next(CommonToken.right_parenthesis)
        return indices, rule

    def _indices(self):
        token = self._next()
        if isinstance(token, int):
            return [token] + self._trailing_indices()
        if token == CommonToken.right_bracket:
            return []
        raise ParseError

    def _trailing_indices(self):
        trailing_indices = []
        while True:
            token = self._next()
            if token == CommonToken.comma:
                index = self._next()
                if not isinstance(index, int):
                    raise ParseError
                trailing_indices.append(index)
            elif token == CommonToken.right_bracket:
                return trailing_indices

    def _formula(self):
        token = self._next()
        if token in [Op.universal, Op.existence, QuantifiedConstant.existential]:
            return token, self._symbol(), self._expr()
        if token == QuantifiedConstant.universal:
            return token, self._symbol()
        if token in [Op.conjunction, Op.disjunction, Op.implication]:
            return token, self._expr(), self._expr()
        if token == Op.negation:
            return token, self._expr()
        if token == Op.contradiction:
            return token
        if isinstance(token, str):
            result = [token]
            while self._peek() != CommonToken.right_parenthesis:
                result.append(self._expr())
            return tuple(result)
        raise ParseError

    def _symbol(self):
        symbol = self._next()
        if isinstance(symbol, str):
            return symbol
        raise ParseError


class ValidationException(Exception):
    pass


@unique
class SubProofKind(Enum):
    arbitrary = 1
    universal = 2
    existential = 3
    conditional = 4

    def __repr__(self):
        return 'SubProofKind.{0}'.format(self.name)


def is_valid_conjunction_introduction(expr, citations):
    """
    Validates the introduction of a conjunction.
    :param expr: The conjunction expression
    :param citations: The lines cited
    :return: the validity of the proof

    >>> P, Q, R, = ('P', 'x'), ('Q', 'x'), ('R', 'x')
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q, R), [])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [None, None, None])
    False
    >>> is_valid_conjunction_introduction((Op.disjunction, P, Q), [None])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [P])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [Q])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, 'p', Q), ['p', Q])
    False
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [P, Q])
    True
    >>> is_valid_conjunction_introduction((Op.conjunction, Q, P), [P, Q])
    True
    >>> is_valid_conjunction_introduction((Op.conjunction, P, Q), [Q, P])
    True
    >>> is_valid_conjunction_introduction((Op.conjunction, P, P), [P])
    True
    """
    if len(expr) != 3 or not (0 < len(citations) <= 2):
        return False
    op, first_arg, second_arg = expr
    if isinstance(first_arg, str) or isinstance(second_arg, str):
        return False
    return op == Op.conjunction and first_arg in citations and second_arg in citations


def is_valid_conjunction_elimination(expr, citations):
    """
    Validates the elimination of a conjunction.
    :param expr: The expression from the conjunction
    :param citations: The conjunction cited
    :return: the validity of the proof

    >>> is_valid_conjunction_elimination(('P', 'x'), [(Op.conjunction, ('P', 'x'), ('Q', 'y')), (Op.conjunction, ('P', 'x'), ('Q', 'y'))])
    False
    >>> is_valid_conjunction_elimination(('P', 'x'), [(Op.conjunction, ('Q', 'x'), ('Q', 'x'))])
    False
    >>> is_valid_conjunction_elimination(('P', 'x'), [(Op.conjunction, ('P', 'x'), ('Q', 'y'))])
    True
    >>> is_valid_conjunction_elimination(('P', 'x'), [(Op.conjunction, ('P', 'x'), ('P', 'x'))])
    True
    >>> is_valid_conjunction_elimination(('P', 'x'), [(Op.conjunction, ('Q', 'y'), ('P', 'x'))])
    True
    """
    if len(citations) != 1:
        return False
    [cited_expr] = citations
    return cited_expr[0] == Op.conjunction and (expr == cited_expr[1] or expr == cited_expr[2])


def is_valid_disjunction_introduction(expr, citations):
    """
    Validates the introduction of a disjunction.
    :param expr: The disjunction expression
    :param citations: The line cited
    :return: the validity of the proof

    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'q', 'r'), [])
    False
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'q'), [])
    False
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'q'), ['p', 'q'])
    False
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'p'), ['p', 'q'])
    False
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'q'), ['p'])
    True
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'q'), ['q'])
    True
    >>> is_valid_disjunction_introduction((Op.disjunction, 'p', 'p'), ['p'])
    True
    """
    if len(expr) != 3 or len(citations) != 1:
        return False
    op, *args = expr
    return op == Op.disjunction and citations[0] in args


def is_valid_disjunction_elimination(expr, citations):
    """
    Validates the elimination of a disjunction.
    :param expr: The expression from the disjunctive elimination
    :param citations: The disjunction and conditionals cited
    :return: the validity of the proof

    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (Op.disjunction, 'p', 'q')])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'q', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.arbitrary, 'q', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.existential, 'q', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.conditional, 'q', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.universal, 'q', ['r'])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'p'), (SubProofKind.conditional, 'p', [])])
    False
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.conditional, 'q', ['r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'p'), (SubProofKind.conditional, 'p', ['r']), (SubProofKind.conditional, 'p', ['r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'p'), (SubProofKind.conditional, 'p', ['r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'p'), (SubProofKind.conditional, 'p', ['s', 'r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(SubProofKind.conditional, 'q', ['r']), (Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'p', ['r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(SubProofKind.conditional, 'p', ['r']), (Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'q', ['r'])])
    True
    >>> is_valid_disjunction_elimination('r', [(Op.disjunction, 'p', 'q'), (SubProofKind.conditional, 'q', ['r']), (SubProofKind.conditional, 'p', ['r'])])
    True
    """
    if len(citations) > 3:
        return False
    conditionals = []
    disjunction = None
    for c in citations:
        tag = c[0]
        if tag == Op.disjunction:
            disjunction = c
        elif tag == SubProofKind.conditional:
            conditionals.append(c)
        else:
            return False
    if disjunction is None or len(disjunction) != 3 or disjunction[0] != Op.disjunction:
        return False
    conditions_used = 0
    for prop in disjunction[1:]:
        if prop not in map(lambda x: x[1], conditionals):
            return False
        conditions_used += 1
    if conditions_used < len(conditionals):
        return False
    return all(map(lambda x: expr in x[2], conditionals))


def is_valid_implication_introduction(expr, citations):
    """
    Validates the introduction of an implication.
    :param expr: The implication introduced
    :param citations: The cited sub-proof
    :return: Whether the introduction is valid

    >>> is_valid_implication_introduction((Op.conjunction, 'p', 'q'), [(SubProofKind.conditional, 'p', ['q'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'r', 'q'), [(SubProofKind.conditional, 'p', ['q'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'q', ['q'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'q', ['p'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'p', ['p'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.universal, 'p', ['q'])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'p', [])])
    False
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'p', ['q'])])
    True
    >>> is_valid_implication_introduction((Op.implication, 'p', 'q'), [(SubProofKind.conditional, 'p', ['q', 'r'])])
    True
    """
    if len(expr) != 3 or len(citations) != 1:
        return False
    [(sub_proof_kind, cited_antecedent, cited_consequents)] = citations
    if sub_proof_kind != SubProofKind.conditional:
        return False
    expr_op, expr_antecedent, expr_consequent = expr
    return expr_op == Op.implication and expr_antecedent == cited_antecedent and expr_consequent in cited_consequents


def is_valid_implication_elimination(expr, citations):
    """
    Validates the elimination of an implication.
    :param expr: The consequent
    :param citations: The proof of the antecedent
    :return: Whether the elimination is valid

    >>> is_valid_implication_elimination('q', [(Op.implication, 'p', 'q'), 'q'])
    False
    >>> is_valid_implication_elimination('q', [(Op.conjunction, 'p', 'q'), 'p'])
    False
    >>> is_valid_implication_elimination('p', [(Op.implication, 'p', 'q'), 'p'])
    False
    >>> is_valid_implication_elimination('q', [(Op.implication, 'p', 'q'), 'p'])
    True
    """
    cited_antecedent = False
    cited_consequent = False
    antecedent = False
    for cited_proof in citations:
        tag, *rest = cited_proof
        if tag == Op.implication:
            cited_antecedent, cited_consequent = rest
        else:
            antecedent = cited_proof
    return cited_antecedent and cited_consequent and antecedent and antecedent == cited_antecedent and expr == cited_consequent


def is_valid_negation_introduction(expr, citations):
    """
    Validates the introduction of a negation.
    :param expr: The negation
    :param citations: The proof of the contradiction arising from an assumption
    :return: Whether the introduction is valid

    >>> is_valid_negation_introduction((Op.negation, 'p'), [(SubProofKind.conditional, 'p', [])])
    False
    >>> is_valid_negation_introduction((Op.negation, 'p'), [(SubProofKind.conditional, 'p', [Op.negation])])
    False
    >>> is_valid_negation_introduction('p', [(SubProofKind.conditional, (Op.negation, 'p'), [Op.contradiction])])
    False
    >>> is_valid_negation_introduction((Op.negation, 'p'), [(SubProofKind.universal, 'p', [Op.contradiction])])
    False
    >>> is_valid_negation_introduction((Op.negation, 'p'), [(SubProofKind.conditional, 'p', [Op.contradiction])])
    True
    >>> is_valid_negation_introduction((Op.negation, 'p'), [(SubProofKind.conditional, 'p', [Op.contradiction, 'q'])])
    True
    """
    if len(expr) != 2:
        return False
    op, negated = expr
    if op != Op.negation:
        return False
    if len(citations) != 1:
        return False
    [subproof] = citations
    if len(subproof) != 3:
        return False
    kind, antecedent, consequents = subproof
    return kind == SubProofKind.conditional and antecedent == negated and Op.contradiction in consequents


def is_valid_negation_elimination(expr, citations):
    """
    Validates the elimination of a negation.
    :param expr: The expression under the double negation.
    :param citations: The double negation.
    :return: Whether the elimination is valid.

    >>> is_valid_negation_elimination('p', [(Op.negation, 'p')])
    False
    >>> is_valid_negation_elimination('q', [(Op.negation, (Op.negation, 'p'))])
    False
    >>> is_valid_negation_elimination('p', ['q', (Op.negation, (Op.negation, 'p'))])
    False
    >>> is_valid_negation_elimination('p', [(Op.negation, (Op.negation, 'p'))])
    True
    >>> is_valid_negation_elimination((Op.negation, 'p'), [(Op.negation, (Op.negation, (Op.negation, 'p')))])
    True
    """
    if len(citations) != 1:
        return False
    [double_negation] = citations
    if len(double_negation) != 2:
        return False
    first_op, first_negated = double_negation
    if first_op != Op.negation or len(first_negated) != 2:
        return False
    second_op, doubly_negated_expr = first_negated
    return second_op == Op.negation and doubly_negated_expr == expr


def is_valid_contradiction_introduction(expr, citations):
    """
    Validates the introduction of a contradiction.
    :param expr: The contradiction.
    :param citations: The proofs that contradict each other.
    :return: Whether the introduction is valid

    >>> is_valid_contradiction_introduction(Op.contradiction, ['a', (Op.negation, 'b')])
    False
    >>> is_valid_contradiction_introduction(Op.contradiction, ['b', (Op.negation, 'a')])
    False
    >>> is_valid_contradiction_introduction(Op.contradiction, ['a', (Op.contradiction, 'a')])
    False
    >>> is_valid_contradiction_introduction(Op.contradiction, ['a', (Op.negation, 'a')])
    True
    >>> is_valid_contradiction_introduction(Op.contradiction, [(Op.negation, 'a'), (Op.negation, (Op.negation, 'a'))])
    True
    """
    if expr != Op.contradiction or len(citations) != 2:
        return False
    [first_citation, second_citation] = citations
    return first_citation == (Op.negation, second_citation) or second_citation == (Op.negation, first_citation)


def is_valid_contradiction_elimination(expr, citations):
    """
    Validates the elimination of a contradiction.
    :param expr: Any expression.
    :param citations: The proof of a contradiction.
    :return: Whether the elimination is valid

    >>> is_valid_contradiction_elimination('a', [Op.contradiction])
    False
    >>> is_valid_contradiction_elimination(('a',), [Op.contradiction])
    True
    >>> is_valid_contradiction_elimination(('P', 'x'), [Op.contradiction])
    True
    >>> is_valid_contradiction_elimination(Op.contradiction, [Op.contradiction])
    True
    """
    return len(citations) == 1 and citations[0] == Op.contradiction and not isinstance(expr, str)


def expr_symbols(expr):
    """
    Computes the set of symbols in an expression.
    :param expr: the expression to scan
    :return: the set of all symbols present in the expression.

    >>> expr_symbols('x') == {'x'}
    True
    >>> expr_symbols(('x',)) == {'x'}
    True
    >>> expr_symbols(('P', 'x')) == {'P', 'x'}
    True
    >>> expr_symbols((Op.universal, 'x', ('P',))) == {'P', 'x'}
    True
    >>> expr_symbols((Op.existence, 'x', ('P',))) == {'P', 'x'}
    True
    >>> expr_symbols((QuantifiedConstant.universal, 'x')) == {'x'}
    True
    >>> expr_symbols((QuantifiedConstant.existential, 'x', ('P',))) == {'P', 'x'}
    True
    >>> expr_symbols((Op.conjunction, ('P',), ('Q',))) == {'P', 'Q'}
    True
    >>> expr_symbols((Op.disjunction, ('P',), ('Q',))) == {'P', 'Q'}
    True
    >>> expr_symbols((Op.implication, ('P',), ('Q',))) == {'P', 'Q'}
    True
    >>> expr_symbols((Op.negation, ('P',))) == {'P'}
    True
    >>> expr_symbols(Op.contradiction) == set()
    True
    """

    result = set()

    def go(sub_expr):
        nonlocal result
        if isinstance(sub_expr, str):
            result.add(sub_expr)
        if sub_expr == Op.contradiction:
            return
        tag = sub_expr[0]
        if tag in [Op.universal, Op.existence, QuantifiedConstant.existential]:
            result.add(sub_expr[1])
            go(sub_expr[2])
        if tag == QuantifiedConstant.universal:
            result.add(sub_expr[1])
        if tag == Op.negation:
            go(sub_expr[1])
        if tag in [Op.conjunction, Op.disjunction, Op.implication]:
            go(sub_expr[1])
            go(sub_expr[2])
        if isinstance(tag, str):
            result.add(tag)
            for e in sub_expr[1:]:
                go(e)

    go(expr)
    return result


def substitute(sym, var, expr):
    """
    Performs capture-avoiding substitution in an expression.
    :param sym: the symbol to substitute in
    :param var: the variable to replace
    :param expr: the expression to be substituted
    :return: a new expression with the substitution

    >>> substitute('a', 'x', ('P', 'a'))
    ('P', 'a')
    >>> substitute('a', 'x', ('P', 'x', 'x'))
    ('P', 'a', 'a')
    >>> substitute('a', 'x', (Op.universal, 'y', ('P', 'x')))
    (Op.universal, 'y', ('P', 'a'))
    >>> substitute('a', 'x', (Op.universal, 'x', ('P', 'x')))
    (Op.universal, 'x', ('P', 'x'))
    >>> substitute('x', 'y', (Op.universal, 'x', ('P', 'x', 'y')))
    (Op.universal, 'Pxy', ('P', 'Pxy', 'x'))
    >>> substitute('a', 'x', (Op.existence, 'y', ('P', 'x')))
    (Op.existence, 'y', ('P', 'a'))
    >>> substitute('a', 'x', (Op.existence, 'x', ('P', 'x')))
    (Op.existence, 'x', ('P', 'x'))
    >>> substitute('x', 'y', (Op.existence, 'x', ('P', 'x', 'y')))
    (Op.existence, 'Pxy', ('P', 'Pxy', 'x'))
    >>> substitute('a', 'x', (Op.conjunction, ('P', 'x'), ('Q', 'y')))
    (Op.conjunction, ('P', 'a'), ('Q', 'y'))
    >>> substitute('a', 'x', (Op.disjunction, ('P', 'x'), ('Q', 'y')))
    (Op.disjunction, ('P', 'a'), ('Q', 'y'))
    >>> substitute('a', 'x', (Op.implication, ('P', 'x'), ('Q', 'y')))
    (Op.implication, ('P', 'a'), ('Q', 'y'))
    >>> substitute('a', 'x', (Op.negation, ('P', 'x')))
    (Op.negation, ('P', 'a'))
    >>> substitute('a', 'x', Op.contradiction)
    Op.contradiction
    """
    if expr == Op.contradiction:
        return expr
    tag = expr[0]
    if isinstance(tag, str):
        result = [tag]
        for obj in expr[1:]:
            result.append(sym if obj == var else obj)
        return tuple(result)
    if tag in [Op.conjunction, Op.disjunction, Op.implication]:
        return tag, substitute(sym, var, expr[1]), substitute(sym, var, expr[2])
    if tag == Op.negation:
        return tag, substitute(sym, var, expr[1])
    if tag in [Op.universal, Op.existence]:
        _, constant, consequent = expr

        if var == constant:
            return expr

        if sym == constant:
            fresh_constant = ''.join(sorted(expr_symbols(sym) | expr_symbols(consequent)))
            consequent = substitute(fresh_constant, constant, consequent)
            constant = fresh_constant

        return tag, constant, substitute(sym, var, consequent)

    raise ValidationException('Unknown tag {0}'.format(tag))


def is_valid_universal_introduction(expr, citations):
    """
    Validates the introduction of a universal proof.
    :param expr: The universal proof
    :param citations: The sub-proof proving the universal statement
    :return: Whether the introduction is valid

    >>> is_valid_universal_introduction((Op.universal, 'x', ('p',)), [(SubProofKind.universal, 'x', [])])
    False
    >>> is_valid_universal_introduction((Op.universal, 'x', ('p',)), [(SubProofKind.universal, 'x', [('q',)])])
    False
    >>> is_valid_universal_introduction((Op.implication, 'x', ('p',)), [(SubProofKind.universal, 'x', [('p',)])])
    False
    >>> is_valid_universal_introduction((Op.universal, 'x', ('p',)), [(SubProofKind.conditional, 'x', [('p',)])])
    False
    >>> is_valid_universal_introduction((Op.universal, 'x', ('P', 'x')), [(SubProofKind.universal, 'y', [('P', 'xy')])])
    False
    >>> is_valid_universal_introduction((Op.universal, ('P', 'x'), ('p',)), [(SubProofKind.universal, 'x', [('p',)])])
    False
    >>> is_valid_universal_introduction((Op.universal, 'x', ('P', 'x')), [(SubProofKind.universal, 'y', [('P', 'x')])])
    False
    >>> is_valid_universal_introduction((Op.universal, 'x', ('p',)), [(SubProofKind.universal, 'x', [('p',)])])
    True
    >>> is_valid_universal_introduction((Op.universal, 'x', ('q',)), [(SubProofKind.universal, 'x', [('p',), ('q',)])])
    True
    >>> is_valid_universal_introduction((Op.universal, 'x', ('P', 'x')), [(SubProofKind.universal, 'y', [('P', 'y')])])
    True
    """
    if len(expr) != 3 or len(citations) != 1:
        return False
    [sub_proof] = citations
    if len(sub_proof) != 3:
        return False
    kind, sub_proof_variable, consequents = sub_proof
    if kind != SubProofKind.universal:
        return False
    op, constant, predicate = expr
    if op != Op.universal or not isinstance(constant, str):
        return False
    return predicate in map(lambda e: substitute(constant, sub_proof_variable, e),
                            filter(lambda e: constant not in expr_symbols(e), consequents))


def is_valid_universal_elimination(expr, citations):
    """
    Validates a universal elimination.
    :param expr: The result of the elimination
    :param citations: The universal statement
    :return: Whether the elimination was valid

    >>> is_valid_universal_elimination(('P', 'y'), [(Op.universal, 'y', ('P', 'x'))])
    False
    >>> is_valid_universal_elimination(('Q', 'x'), [(Op.universal, 'y', ('P', 'y'))])
    False
    >>> is_valid_universal_elimination(('x',), [(Op.universal, 'y', ('x',))])
    True
    >>> is_valid_universal_elimination(('P', 'y'), [(Op.universal, 'y', ('P', 'y'))])
    True
    >>> is_valid_universal_elimination(('P', 'x'), [(Op.universal, 'y', ('P', 'y'))])
    True
    """
    if len(citations) != 1 or isinstance(expr, str):
        return False
    [citation] = citations
    if len(citation) != 3:
        return False
    op, constant, predicate = citation
    if op != Op.universal:
        return False
    return expr in (substitute(sym, constant, predicate) for sym in expr_symbols(expr))


def is_valid_existential_introduction(proof, citations):
    """
    Validates an existential introduction.
    :param proof: The existential claim.
    :param citations: The specific claim being generalized.
    :return: Whether the introduction is valid

    >>> is_valid_existential_introduction((Op.existence, 'x', ('P', 'x')), [('P', 'x')])
    False
    >>> is_valid_existential_introduction((Op.existence, 'x', ('P', 'y', 'x')), [('P', 'z', 'y')])
    False
    >>> is_valid_existential_introduction((Op.universal, 'x', ('P', 'x')), [('P', 'y')])
    False
    >>> is_valid_existential_introduction((Op.existence, 'x', ('P', 'x')), [])
    False
    >>> is_valid_existential_introduction((Op.existence, 'x', ('P', 'x')), [('P', 'y')])
    True
    >>> is_valid_existential_introduction((Op.existence, 'x', ('P', 'x', 'y')), [('P', 'z', 'y')])
    True
    """
    if len(citations) != 1 or len(proof) != 3:
        return False
    op, constant, predicate = proof
    if op != Op.existence:
        return False
    [expr] = citations
    return constant not in expr_symbols(expr) and predicate in (substitute(constant, var, expr) for var in
                                                                expr_symbols(expr))


def is_valid_existential_elimination(proof, citations):
    """
    Validates an existential elimination.
    :param proof: The result of the sub-proof.
    :param citations: The sub-proof proving the result.
    :return: Whether the elimination is valid

    >>> is_valid_existential_elimination(('p', 'x'), [(SubProofKind.existential, 'x', [('p', 'x'), ('p', 'y')])])
    False
    >>> is_valid_existential_elimination(('p',), [(SubProofKind.universal, 'x', [('p',)])])
    False
    >>> is_valid_existential_elimination(('p',), [(SubProofKind.existential, 'x', [('p',)])])
    True
    >>> is_valid_existential_elimination(('p', 'y'), [(SubProofKind.existential, 'x', [('p', 'x'), ('p', 'y')])])
    True
    """
    if len(citations) != 1:
        return False
    [sub_proof] = citations
    if len(sub_proof) != 3:
        return False
    kind, constant, predicates = sub_proof
    return kind == SubProofKind.existential and proof in predicates and constant not in expr_symbols(proof)


def is_valid_reiteration(proof, citations):
    """
    Validates a reiteration.
    :param proof: The reiterated proof.
    :param citations: The original proof.
    :return: Whether the reiteration is valid

    >>> is_valid_reiteration('x', ['y'])
    False
    >>> is_valid_reiteration('x', [])
    False
    >>> is_valid_reiteration('x', ['x'])
    True
    >>> is_valid_reiteration('x', ['x', 'y'])
    True
    """
    return proof in citations


def is_valid_line(line_number, expr, cited_line_numbers, rule, context, symbols):
    """
    Validates the line, given the current context and symbol set.
    :param line_number: the index of the line
    :param expr: the expression being proved
    :param cited_line_numbers: the lines referred to
    :param rule: the inference rule used
    :param context: dictionary of previous proofs by line number
    :param symbols: set of previous symbols
    :return: whether the line is valid

    >>> is_valid_line(10, ('x',), [], InferenceRule.supposition, {10: ('y',)}, {'y'})
    False
    >>> is_valid_line(10, 'x', [], InferenceRule.supposition, {5: ('y',)}, {'y'})
    False
    >>> is_valid_line(10, ('x',), [], InferenceRule.supposition, {5: ('x',)}, {'x'})
    False
    >>> is_valid_line(10, ('x',), [5], InferenceRule.supposition, {5: ('x',)}, {'x'})
    False
    >>> is_valid_line(10, ('x',), [], InferenceRule.supposition, {5: ('y',)}, {'x', 'y'})
    True
    >>> is_valid_line(10, (QuantifiedConstant.universal, 'x'), [5], QuantifiedConstant.universal, {5: ('x',)}, {'x'})
    False
    >>> is_valid_line(10, (QuantifiedConstant.universal, 'x'), [], QuantifiedConstant.universal, {5: ('x',)}, {'x'})
    False
    >>> is_valid_line(10, (QuantifiedConstant.universal, 'x'), [], QuantifiedConstant.universal, {}, set())
    True
    >>> is_valid_line(10, (QuantifiedConstant.existential, 'a', ('P', 'a')), [5], QuantifiedConstant.existential,{5: (Op.existence, 'x', ('P', 'x'))}, {'a', 'x', 'P'})
    False
    >>> is_valid_line(10, (QuantifiedConstant.existential, 'a', ('P', 'a')), [5], QuantifiedConstant.existential,{5: (Op.existence, 'x', ('P', 'x'))}, {'x', 'P'})
    True
    >>> is_valid_line(20, (Op.conjunction, ('Q', 'a'), ('P', 'a')), [5, 10], InferenceRule.conjunction_introduction,{5: ('Q', 'a'), 10: ('P', 'a')}, {'a', 'Q', 'P'})
    True
    >>> is_valid_line(20, ('Q', 'a'), [10], InferenceRule.conjunction_elimination,{10: (Op.conjunction, ('Q', 'a'), ('P', 'a'))}, {'a', 'Q', 'P'})
    True
    >>> is_valid_line(20, (Op.disjunction, ('Q', 'a'), ('P', 'a')), [5], InferenceRule.disjunction_introduction,{5: ('Q', 'a')}, {'a', 'Q', 'P'})
    True
    """
    if line_number in context or isinstance(expr, str):
        return False

    if rule == InferenceRule.supposition:
        return len(cited_line_numbers) == 0 and expr not in context.values()

    if rule == QuantifiedConstant.universal:
        if len(cited_line_numbers) != 0 or len(expr) != 2:
            return False
        quantifier, sym = expr
        return quantifier == QuantifiedConstant.universal and sym not in symbols

    if rule == QuantifiedConstant.existential:
        if len(cited_line_numbers) != 1 or len(expr) != 3:
            return False
        [cited_line] = cited_line_numbers
        if cited_line not in context:
            return False
        quantifier, sym, prop = expr
        if quantifier != QuantifiedConstant.existential or sym in symbols:
            return False
        citation = context[cited_line]
        if len(citation) != 3:
            return False
        cited_quantifier, var, ex_prop = citation
        return cited_quantifier == Op.existence and substitute(sym, var, ex_prop) == prop

    citations = []
    for cited_index in cited_line_numbers:
        if cited_index not in context:
            return False
        citations.append(context[cited_index])

    if rule == InferenceRule.conjunction_introduction:
        return is_valid_conjunction_introduction(expr, citations)
    if rule == InferenceRule.conjunction_elimination:
        return is_valid_conjunction_elimination(expr, citations)
    if rule == InferenceRule.disjunction_introduction:
        return is_valid_disjunction_introduction(expr, citations)
    if rule == InferenceRule.disjunction_elimination:
        return is_valid_disjunction_elimination(expr, citations)
    if rule == InferenceRule.implication_introduction:
        return is_valid_implication_introduction(expr, citations)
    if rule == InferenceRule.implication_elimination:
        return is_valid_implication_elimination(expr, citations)
    if rule == InferenceRule.negation_introduction:
        return is_valid_negation_introduction(expr, citations)
    if rule == InferenceRule.negation_elimination:
        return is_valid_negation_elimination(expr, citations)
    if rule == InferenceRule.universal_introduction:
        return is_valid_universal_introduction(expr, citations)
    if rule == InferenceRule.universal_elimination:
        return is_valid_universal_elimination(expr, citations)
    if rule == InferenceRule.existential_introduction:
        return is_valid_existential_introduction(expr, citations)
    if rule == InferenceRule.existential_elimination:
        return is_valid_existential_elimination(expr, citations)
    if rule == InferenceRule.contradiction_introduction:
        return is_valid_contradiction_introduction(expr, citations)
    if rule == InferenceRule.contradiction_elimination:
        return is_valid_contradiction_elimination(expr, citations)
    if rule == InferenceRule.reiteration:
        return is_valid_reiteration(expr, citations)

    raise ValidationException('Unknown rule {0}'.format(rule))


def validate_proof(proof, context, symbols):
    if len(proof) == 3:
        line_number, expr, (cited_line_numbers, rule) = proof
        if not is_valid_line(line_number, expr, cited_line_numbers, rule, context, symbols):
            return None
        if rule == InferenceRule.supposition:
            context = context.copy()
            context[line_number] = expr
            return context, symbols | expr_symbols(expr)
        if rule == QuantifiedConstant.universal:
            return context, symbols | expr[1]
        if rule == QuantifiedConstant.existential:
            context = context.copy()
            context[line_number] = expr[2]
            return context, symbols | expr[1]
        return context, symbols

    proof_line_number, sub_proof_list = proof

    if proof_line_number in context:
        return None

    sub_proof_kind = SubProofKind.arbitrary

    inner_context = context
    inner_symbols = symbols
    for sub_proof in sub_proof_list:
        inner_context, inner_symbols = validate_proof(sub_proof, inner_context, inner_symbols)
        if len(sub_proof) == 3:
            _, _, (_, rule) = sub_proof
            if rule == InferenceRule.supposition:
                if sub_proof_kind != SubProofKind.arbitrary:
                    return None
                sub_proof_kind = SubProofKind.conditional
            if rule == QuantifiedConstant.universal:
                if sub_proof_kind != SubProofKind.arbitrary:
                    return None
                sub_proof_kind = SubProofKind.universal
            if rule == QuantifiedConstant.existential:
                if sub_proof_kind != SubProofKind.arbitrary:
                    return None
                sub_proof_kind = SubProofKind.existential

    consequents = [prop for line_number, prop in inner_context if line_number not in context]
    context = context.copy()
    context[proof_line_number] = (sub_proof_kind, consequents)
    return context, symbols


def verify(proof_str):
    context, _ = validate_proof(Parser().parse(proof_str), dict(), set())
    [sub_proof_kind] = context.keys()
    return 'V' if sub_proof_kind == SubProofKind.conditional else 'I'


# noinspection PyPep8Naming
def verifyProof(P):
    """
    :param P: A string which is an S-expression of a well-formed Fitch-style
    proof.
    :return: Returns either:
        “I” – If P was well-formed, but not a valid proof,
        “V” – If P was well-formed, and a valid proof.
    """
    # noinspection PyBroadException
    try:
        return verify(P)
    except Exception:
        return 'I'
