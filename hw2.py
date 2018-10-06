"""
Created by Spencer Turkel on 09/27/2018.
"""
import re
from enum import Enum, unique
from string import whitespace


@unique
class QuantifiedConstant(Enum):
    universal_constant = 1
    existential_constant = 2

    def __repr__(self):
        return 'QuantifiedConstant.{0}'.format(self.name)


@unique
class Op(Enum):
    universal = 1
    existence = 2
    contradiction = 3
    conjunction = 4
    disjunction = 5
    implication = 6
    negation = 7

    def __repr__(self):
        return 'Op.{0}'.format(self.name)


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
    reiteration = 16

    def __repr__(self):
        return 'InferenceRule.{0}'.format(self.name)


@unique
class CommonToken(Enum):
    left_parenthesis = 1
    right_parenthesis = 2
    left_bracket = 3
    right_bracket = 4
    comma = 5
    sub_proof = 6

    def __repr__(self):
        return 'CommonToken.{0}'.format(self.name)


class InputSyntaxError(Exception):
    pass


class Lexer:
    """
    An iterable class for lexing.
    You can also peek at the next character (excluding whitespace).

    >>> list(Lexer('az,() 123'))
    ['az', CommonToken.comma, CommonToken.left_parenthesis, CommonToken.right_parenthesis, 123]
    >>> list(Lexer('XE RE S'))
    [InferenceRule.contradiction_elimination, InferenceRule.reiteration, InferenceRule.supposition]
    >>> list(Lexer(' [\t] \t  0   ab123'))
    [CommonToken.left_bracket, CommonToken.right_bracket, 0, 'ab123']
    >>> l = Lexer(' [\t] \t  0   ab123')
    >>> l.peek()
    CommonToken.left_bracket
    >>> next(l)
    CommonToken.left_bracket
    >>> l.peek()
    CommonToken.right_bracket
    >>> next(l)
    CommonToken.right_bracket
    >>> l.peek()
    0
    >>> next(l)
    0
    >>> l.peek()
    'ab123'
    >>> next(l)
    'ab123'
    >>> l.peek()
    >>> next(l)
    Traceback (most recent call last):
        ...
    StopIteration
    >>> l = Lexer('Abc')
    >>> l.peek()
    'Abc'
    >>> next(l)
    'Abc'
    """

    # regular expressions compiled for lexing
    _index_regex = re.compile(r'\d+')
    _object_regex = re.compile('[A-Za-z0-9]+')

    def __init__(self, text):
        self.length = len(text)
        self.index = 0
        self.text = text

    def __iter__(self):
        return self

    def peek(self):
        """
        Peeks at the next lexeme in the text.
        :return: The next lexeme in the text, or None if there are none.
        """
        if self.index == self.length:
            return None
        next_char = self._next_char()
        if next_char == '(':
            return CommonToken.left_parenthesis
        if next_char == ')':
            return CommonToken.right_parenthesis
        if next_char == '[':
            return CommonToken.left_bracket
        if next_char == ']':
            return CommonToken.right_bracket
        if next_char == ',':
            return CommonToken.comma
        if self._next_word_is('SUBP'):
            return CommonToken.sub_proof
        if self._next_word_is('UCONST'):
            return QuantifiedConstant.universal_constant
        if self._next_word_is('ECONST'):
            return QuantifiedConstant.existential_constant
        if self._next_word_is('FORALL'):
            return Op.universal
        if self._next_word_is('EXISTS'):
            return Op.existence
        if self._next_word_is('CONTR'):
            return Op.contradiction
        if self._next_word_is('AND'):
            return Op.conjunction
        if self._next_word_is('OR'):
            return Op.disjunction
        if self._next_word_is('IMPLIES'):
            return Op.implication
        if self._next_word_is('NOT'):
            return Op.negation
        if self._next_word_is('S'):
            return InferenceRule.supposition
        if self._next_word_is('CI'):
            return InferenceRule.conjunction_introduction
        if self._next_word_is('CE'):
            return InferenceRule.conjunction_elimination
        if self._next_word_is('DI'):
            return InferenceRule.disjunction_introduction
        if self._next_word_is('DE'):
            return InferenceRule.disjunction_elimination
        if self._next_word_is('II'):
            return InferenceRule.implication_introduction
        if self._next_word_is('IE'):
            return InferenceRule.implication_elimination
        if self._next_word_is('NI'):
            return InferenceRule.negation_introduction
        if self._next_word_is('NE'):
            return InferenceRule.negation_elimination
        if self._next_word_is('AI'):
            return InferenceRule.universal_introduction
        if self._next_word_is('AE'):
            return InferenceRule.universal_elimination
        if self._next_word_is('EI'):
            return InferenceRule.existential_introduction
        if self._next_word_is('EE'):
            return InferenceRule.existential_elimination
        if self._next_word_is('XI'):
            return InferenceRule.contradiction_introduction
        if self._next_word_is('XE'):
            return InferenceRule.contradiction_elimination
        if self._next_word_is('RE'):
            return InferenceRule.reiteration
        match = self._next_word_match(Lexer._index_regex)
        if match:
            return int(match.group(0))
        match = self._next_word_match(Lexer._object_regex)
        return match.group(0) if match else None

    def _next_char(self):
        while True:
            if self.index == self.length:
                raise StopIteration
            ch = self.text[self.index]
            if ch not in whitespace:
                return ch
            self.index += 1

    def _next_word_is(self, word):
        return self.text.startswith(word, self.index)

    def _next_word_match(self, regex):
        return regex.match(self.text, self.index)

    def __next__(self):
        if self.index == self.length:
            raise StopIteration
        next_char = self._next_char()
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
        if self._next_word_is('RE'):
            self.index += len('RE')
            return InferenceRule.reiteration
        match = self._next_word_match(Lexer._index_regex)
        if match:
            index_lexeme = match.group(0)
            self.index += len(index_lexeme)
            return int(index_lexeme)
        match = self._next_word_match(Lexer._object_regex)
        if match:
            object_lexeme = match.group(0)
            self.index += len(object_lexeme)
            return object_lexeme
        raise InputSyntaxError


class ParseError(Exception):
    pass


class Parser:
    def __init__(self):
        self._scanner = None

    def parse(self, text):
        """
        This is a recursive descent parser for the input lexemes, returning a structured Proof.

        :param text: a Fitch-style proof (see README or assignment requirements for format).
        :return: an element of type Proof (defined below).
        :except ParseError: when the proof cannot be parsed.

        Result type::

            Proof = Union[Tuple[int, List[Proof]], Tuple[int, Expr, Justification]]
            Expr = Union[ str,
                          Tuple[Op.universal,  str,  Expr],
                          Tuple[Op.existence,  str,  Expr],
                          Tuple[QuantifiedConstant.universal_constant,  str],
                          Tuple[QuantifiedConstant.existential_constant,  str,  Expr],
                          Tuple[Op.conjunction,  Expr,  Expr],
                          Tuple[Op.disjunction,  Expr,  Expr],
                          Tuple[Op.implication,  Expr,  Expr],
                          Tuple[Op.negation,  Expr],
                          Op.contradiction,
                          Tuple[str,  Expr],
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
        (10, (QuantifiedConstant.universal_constant, 'p'), ([], QuantifiedConstant.universal_constant))
        >>> p.parse('(10 (ECONST p (q p)) ([] ECONST))')
        (10, (QuantifiedConstant.existential_constant, 'p', ('q', 'p')), ([], QuantifiedConstant.existential_constant))
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
        if token in [Op.universal, Op.existence, QuantifiedConstant.existential_constant]:
            return token, self._symbol(), self._expr()
        if token == QuantifiedConstant.universal_constant:
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
    # try:
    #     validate(Parser().parse(P))
    #     return 'V'
    # except ValidationException or ParseError:
    #     return 'I'
