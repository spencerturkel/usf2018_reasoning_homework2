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
        ('a',
         ('FORALL', 'x', ('FORALL', 'a', ('FORALL', '_FRESH_1', ('P', 'a', '_FRESH_1')))),
         ('FORALL', '_FRESH_1', ('FORALL', '_FRESH_2', ('P', '_FRESH_1', '_FRESH_2')))),
        ('y',
         ('FORALL', 'x', ('EXISTS', 'y', ('P', 'x', 'y'))),
         ('EXISTS', '_FRESH_1', ('P', 'y', '_FRESH_1'))),
        ('a',
         ('FORALL', 'x', ('AND', ('P', 'x'), ('EXISTS', 'a', ('P', 'a')))),
         ('AND', ('P', 'a'), ('EXISTS', '_FRESH_1', ('P', '_FRESH_1')))),
        ('a', ('EXISTS', 'x', ('P', 'x')), ('P', 'a')),
        ('a', ('EXISTS', 'a', ('P', 'a')), ('P', 'a')),
        ('a',
         ('EXISTS', 'x', ('AND', ('P', 'x'), ('FORALL', 'a', ('P', 'a')))),
         ('AND', ('P', 'a'), ('FORALL', '_FRESH_1', ('P', '_FRESH_1')))),
    ])
    def test_good(obj, quantifier_predicate, result):
        fresh_count = 0

        def fresh():
            nonlocal fresh_count
            fresh_count += 1
            return '_FRESH_{}'.format(fresh_count)

        assert instantiate(obj, quantifier_predicate, fresh) == result

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
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x'), dict(), set(), set(), set()),
                ((10, 'x'), {5: 'y'}, {'P', 'Q'}, {'f', 'g'}, {'y', 'z'}),
            ]
        )
        def test_good_universal_constant(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            _, variable = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs - {variable}

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x'), dict(), {'x'}, set(), set()),
                ((10, 'x'), dict(), set(), {'x'}, set()),
                ((10, 'x'), dict(), set(), set(), {'x'}),
                ((10, 'x'), {10: 'y'}, set(), set(), {'y'}),
            ]
        )
        def test_bad_universal_constant(proof, facts, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts, seen_predicates, seen_functions, seen_objects)

    class TestExistentialConstant:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x', 'CONTR', 5), {5: ('EXISTS', 'y', 'CONTR')}, set(), set(), {'y'}),
                ((10, 'x', ('P', 'x'), 5), {5: ('EXISTS', 'y', ('P', 'y'))}, {'P'}, set(), {'y'}),
                ((10, 'x', ('P', 'x', 'z'), 5), {5: ('EXISTS', 'y', ('P', 'y', 'z'))},
                 {'P'}, set(), {'y', 'z'}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, variable, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != proof[0]}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs - {variable}

        @staticmethod
        @pytest.mark.parametrize(
            'proof, seen_predicates, seen_functions, seen_objects', [
                ((10, 'x', ('P', 'x'), 5), set(), set(), {'x', 'y'}),
                ((10, 'x', ('Q', 'x'), 5), {'P', 'Q'}, set(), {'y'}),
                ((10, 'f', ('P', 'f'), 5), {'P'}, {'f'}, {'y'}),
            ])
        def test_bad(proof, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, {5: ('EXISTS', 'y', ('P', 'y'))},
                               seen_predicates, seen_functions, seen_objects)

    class TestConjunctionIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P',), ('Q',)), [40, 30], 'CI'),
                 {30: ('P',), 40: ('Q',)},
                 {'P', 'Q'}, set(), set()),
                ((50, ('AND', ('P',), ('P',)), [30], 'CI'),
                 {30: ('P',)},
                 {'P', 'Q'}, set(), set()),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != proof[0]}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30], 'CI'),
                 {30: ('P', ('f', 'x'))}, {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [40], 'CI'),
                 {40: ('Q', 'y')}, {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30, 40, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q',)), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('AND', ('P',), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates, seen_functions, seen_objects)

    class TestConjunctionElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('P',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))},
                 {'P', 'Q'}, set(), set()),
                ((50, ('Q',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))},
                 {'P', 'Q'}, set(), set()),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != proof[0]}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('P',), [], 'CE'), dict(), {'P', 'Q'}, set(), set()),
                ((50, ('P',), [40, 40], 'CE'), {40: ('AND', ('P',), ('Q',))},
                 {'P', 'Q'}, set(), set()),
                ((50, ('R',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))},
                 {'P', 'Q', 'R'}, set(), set()),
                ((50, ('P',), [40], 'CE'), {40: ('OR', ('P',), ('Q',))},
                 {'P', 'Q'}, set(), set()),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates, seen_functions, seen_objects)

    class TestDisjunctionIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != proof[0]}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, seen_predicates, seen_functions, seen_objects', [
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P'}, {'f'}, {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P', 'Q'}, {'f'}, {'x'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P', 'Q'}, set(), {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40, 30], 'DI'),
                 {40: ('P', ('f', 'x')), 30: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q',)), [40], 'DI'),
                 {40: ('Q', 'y')},
                 {'P', 'Q'}, {'f'}, {'x', 'y'}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates, seen_functions, seen_objects)

    class TestDisjunctionElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',), ('S',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',), ('T',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {30: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  20: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {40: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  20: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  40: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  30: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: ('OR', ('P',), ('P',))}),
            ])
        def test_good(proof, facts_by_index):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       set(), set(), set())
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert set() == preds == funcs == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',), ('S',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',), ('S',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('S',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('S',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('S',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('AND', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.universal, ('P',), {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q', 'x')}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [20, 30], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('R',), [30, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
                ((50, ('P',), [30], 'DE'),
                 {30: ('OR', ('P',), ('P',))}),
                ((50, ('R',), [20, 30, 40, 40], 'DE'),
                 {20: (SubProofKind.conditional, {('P',)}, {('R',)}),
                  30: (SubProofKind.conditional, {('Q',)}, {('R',)}),
                  40: ('OR', ('P',), ('Q',))}),
            ])
        def test_bad(proof, facts_by_index):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, set(), set(), set())

    class TestImplicationIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q',)})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q',), ('R',)})}),
                ((50, ('IMPLIES', ('P', 'x'), ('Q', 'y')), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {('Q', 'y')})}),
            ])
        def test_good(proof, facts_by_index):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       set(), set(), set())
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert set() == preds == funcs == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('AND', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q',)})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P',), ('R',)}, {('Q',)})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.universal, {('P',)}, {('Q',)})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {('Q',)})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q', 'x')})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q', 'x')})}),
                ((50, ('IMPLIES', ('P',), ('Q',)), [20, 20], 'II'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q', 'x')})}),
            ])
        def test_bad(proof, facts_by_index):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, set(), set(), set())

    @pytest.mark.skip
    class TestImplicationElimination:
        pass

    @pytest.mark.skip
    class TestNegationIntroduction:
        pass

    @pytest.mark.skip
    class TestNegationElimination:
        pass

    @pytest.mark.skip
    class TestContradictionIntroduction:
        pass

    @pytest.mark.skip
    class TestContradictionElimination:
        pass

    @pytest.mark.skip
    class TestUniversalIntroduction:
        pass

    @pytest.mark.skip
    class TestUniversalElimination:
        pass

    @pytest.mark.skip
    class TestExistenceIntroduction:
        pass

    @pytest.mark.skip
    class TestExistenceElimination:
        pass

    @pytest.mark.skip
    class TestSupposition:
        pass

    @pytest.mark.skip
    class TestReiteration:
        pass

    @pytest.mark.skip
    class TestArbitrarySubProof:
        pass

    @pytest.mark.skip
    class TestConditionalSubProof:
        pass

    @pytest.mark.skip
    class TestUniversalSubProof:
        pass

    @pytest.mark.skip
    class TestExistentialSubProof:
        pass
