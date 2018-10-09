from hw2 import *

import pytest


class TestLexer:
    @staticmethod
    @pytest.mark.parametrize(['string', 'result'], [
        ('UCONST ECONST', ['UCONST', 'ECONST']),
        ('FORALL EXISTS CONTR AND OR IMPLIES NOT', ['FORALL', 'EXISTS', 'CONTR',
                                                    'AND', 'OR', 'IMPLIES', 'NOT']),
        ('S CI CE DI DE II IE NI NE AI AE EI EE XI XE RE', ['S', 'CI', 'CE', 'DI',
                                                            'DE', 'II', 'IE', 'NI',
                                                            'NE', 'AI', 'AE', 'EI',
                                                            'EE', 'XI', 'XE', 'RE']),
        ('()[],SUBP', ['(', ')', '[', ']', ',', 'SUBP']),
        ('AND ORr', ['AND', 'ORr']),
        ('ANDpq OR', ['ANDpq', 'OR']),
        ('az,() 123', ['az', ',', '(', ')', 123]),
        ('(10 (AND (x) (y)) ([] S))', ['(', 10, '(', 'AND', '(', 'x', ')', '(', 'y',
                                       ')', ')', '(', '[', ']', 'S', ')', ')']),
        (' [\t] \n\t  0   ab123', ['[', ']', 0, 'ab123'])
    ])
    def test_listing(string, result):
        assert list(Lexer(string)) == result

    @staticmethod
    def test_peeking_and_nexting():
        lexer = Lexer(' [\t] \t  0   Ab123')
        for token in ['[', ']', 0, 'Ab123']:
            assert lexer.peek() == token
            assert next(lexer) == token
        assert lexer.peek() is None
        with pytest.raises(StopIteration):
            next(lexer)


class ListLexer:
    """
    An iterable returning user-supplied values in order, and allowing a .peek() operation.
    """

    def __init__(self, values):
        self.values = values

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.values.pop(0)
        except IndexError:
            raise StopIteration

    def peek(self):
        return self.values[0] if len(self.values) > 0 else None


def test_list_lexer():
    lexer = ListLexer([1, 2, 3])
    assert 1 == lexer.peek()
    assert next(lexer) == 1
    assert 2 == lexer.peek()


class TestParse:
    @staticmethod
    @pytest.mark.parametrize('ast,lexer', [
        ((10, []), ListLexer(['(', 'SUBP', 10, ')'])),
        ((10, 'x'), ListLexer(['(', 10,
                               '(', 'UCONST', 'x', ')',
                               '(', '[', ']', 'UCONST', ')', ')'])),
        ((10, 'x', ('P',), 5), ListLexer(['(', 10,
                                          '(', 'ECONST', 'x', '(', 'P', ')', ')',
                                          '(', '[', 5, ']', 'ECONST', ')', ')'])),
        ((10, ('P',), [], 'S'), ListLexer(['(', 10,
                                           '(', 'P', ')',
                                           '(', '[', ']', 'S', ')', ')'])),
    ])
    def test_special_lines(ast, lexer):
        assert ast == parse(lexer)

    @staticmethod
    @pytest.mark.parametrize('rule',
                             ['S', 'CI', 'CE', 'DI', 'DE',
                              'II', 'IE', 'NI', 'NE', 'AI',
                              'AE', 'EI', 'EE', 'XI', 'XE', 'RE'])
    def test_rules(rule):
        line_tokens = ['(', 10,
                       '(', 'P', ')',
                       '(', '[', ']', rule, ')', ')']
        assert (10, ('P',), [], rule) == parse(ListLexer(line_tokens))

    @staticmethod
    @pytest.mark.parametrize('predicate, tokens', [
        (('FORALL', 'x', ('P',)), ['FORALL', 'x', '(', 'P', ')']),
        (('EXISTS', 'x', ('P',)), ['EXISTS', 'x', '(', 'P', ')']),
        (('AND', ('P',), ('Q',)), ['AND', '(', 'P', ')', '(', 'Q', ')']),
        (('OR', ('P',), ('Q',)), ['OR', '(', 'P', ')', '(', 'Q', ')']),
        (('IMPLIES', ('P',), ('Q',)), ['IMPLIES', '(', 'P', ')', '(', 'Q', ')']),
        (('NOT', ('P',)), ['NOT', '(', 'P', ')']),
        (('P', 'x', ('Q', 'y'), 'z'), ['P', 'x', '(', 'Q', 'y', ')', 'z']),
        ('CONTR', ['CONTR']),
    ])
    def test_predicates(predicate, tokens):
        line_tokens = ['(', 10, '(',
                       ] + tokens + [
                          ')', '(', '[', ']', 'S', ')', ')']
        assert (10, predicate, [], 'S') == parse(ListLexer(line_tokens))

    @staticmethod
    @pytest.mark.parametrize('index', [
        10, 5, 0, 101, 15
    ])
    def test_line_indices(index):
        subproof_tokens = ['(', 'SUBP', index, ')']
        line_tokens = ['(', index,
                       '(', 'P', ')',
                       '(', '[', ']', 'S', ')', ')']
        assert (index, []) == parse(ListLexer(subproof_tokens))
        assert (index, ('P',), [], 'S') == parse(ListLexer(line_tokens))

    @staticmethod
    @pytest.mark.parametrize('citations, tokens', [
        ([], []),
        ([5, 10, 15], [5, ',', 10, ',', 15]),
        ([0], [0])
    ])
    def test_citations(citations, tokens):
        line_tokens = ['(', 10,
                       '(', 'P', ')',
                       '(', '['] + tokens + [']', 'S', ')', ')']
        assert (10, ('P',), citations, 'S') == parse(ListLexer(line_tokens))

    @staticmethod
    @pytest.mark.parametrize('lines, tokens', [
        ([], []),
        ([(10, ('P',), [], 'S')], ['(', 10,
                                   '(', 'P', ')',
                                   '(', '[', ']', 'S', ')', ')']),
        ([(10, ('P',), [], 'S'),
          (20, ('Q', 'x'), [], 'S')], [
             '(', 10,
             '(', 'P', ')',
             '(', '[', ']', 'S', ')', ')',
             '(', 20,
             '(', 'Q', 'x', ')',
             '(', '[', ']', 'S', ')', ')',
         ]),
    ])
    def test_sub_proofs(lines, tokens):
        line_tokens = ['(', 'SUBP', 10] + tokens + [')']
        assert (10, lines) == parse(ListLexer(line_tokens))


@pytest.mark.parametrize('symbols, predicate', [
    (({'P'}, set(), set()), ('P',)),
    (({'P'}, set(), {'x', 'y'}), ('P', 'x', 'y')),
    (({'P'}, {'f'}, {'x'}), ('P', ('f', 'x'))),
    (({'P', 'Q'}, {'f', 'g'}, {'x', 'y'}), ('AND', ('P', ('f', 'x', 'y')), ('Q', ('g', 'x')))),
    (({'P', 'Q'}, {'f'}, {'y'}), ('NOT', ('P', ('f', ('NOT', ('Q',)), 'y')))),
    (({'P', 'Q'}, set(), {'y'}), ('NOT', ('P', ('NOT', ('Q',)), 'y'))),
])
def test_symbols_of(symbols, predicate):
    assert symbols == symbols_of(predicate)


class TestInstantiate:
    @staticmethod
    @pytest.mark.parametrize('obj, quantifier_predicate, result', [
        ('a', ('FORALL', 'x', ('P', 'x')), ('P', 'a')),
        ('a', ('FORALL', 'a', ('P', 'a')), ('P', 'a')),
        ('y',
         ('FORALL', 'x', ('EXISTS', 'y', ('P', 'x', 'y'))),
         ('EXISTS', 'y', '_FRESH')),
        ('a',
         ('FORALL', 'x', ('AND', ('P', 'x'), ('EXISTS', 'a', ('P', 'a')))),
         ('AND', ('P', 'a'), ('EXISTS', 'a', ('P', 'a')))),
        ('a', ('EXISTS', 'x', ('P', 'x')), ('P', 'a')),
        ('a', ('EXISTS', 'a', ('P', 'a')), ('P', 'a')),
        ('a',
         ('EXISTS', 'x', ('AND', ('P', 'x'), ('FORALL', 'a', ('P', 'a')))),
         ('AND', ('P', 'a'), ('FORALL', 'a', ('P', 'a')))),
    ])
    def test_good(obj, quantifier_predicate, result):
        assert instantiate(obj, quantifier_predicate, lambda: '_FRESH') == result

    @staticmethod
    @pytest.mark.parametrize('obj, quantifier_predicate', [
        ('a', ('P', 'x')),
        ('a', 'x'),
        ('a', (SubProofKind.conditional, set(), {('FORALL', 'x', ('P', 'x'))})),
        ('a', (SubProofKind.universal, 'y', {('FORALL', 'x', ('P', 'x'))})),
        ('a', (SubProofKind.existential, 'y', {('FORALL', 'x', ('P', 'x'))})),
    ])
    def test_bad(obj, quantifier_predicate):
        with pytest.raises(InvalidProof):
            instantiate(obj, quantifier_predicate, lambda: '_FRESH')


class TestValidateProof:
    class TestUniversalConstant:
        @staticmethod
        @pytest.mark.parametrize(
            'proof, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x'), set(), set(), set()),
                ((10, 'x'), {'P', 'Q'}, {'f', 'g'}, {'y', 'z'}),
            ]
        )
        def test_good_universal_constant(proof, seen_predicates, seen_functions, seen_objects):
            validate_proof(proof, dict(), seen_predicates, seen_functions, seen_objects)

        @staticmethod
        @pytest.mark.parametrize(
            'proof, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x'), {'x'}, set(), set()),
                ((10, 'x'), set(), {'x'}, set()),
                ((10, 'x'), set(), set(), {'x'}),
            ]
        )
        def test_bad_universal_constant(proof, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, dict(), seen_predicates, seen_functions, seen_objects)

    @pytest.mark.skip
    class TestExistentialConstant:
        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_line, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x', 'CONTR', 5), {5: {('EXISTS', 'y', 'CONTR')}}, set(), set(), {'y'}),
                ((10, 'x', ('P', 'x'), 5), {5: {('EXISTS', 'y', ('P', 'y'))}}, {'P'}, set(), {'y'}),
            ]
        )
        def test_good(proof, facts_by_line, seen_predicates, seen_functions, seen_objects):
            validate_proof(proof, facts_by_line, seen_predicates, seen_functions, seen_objects)