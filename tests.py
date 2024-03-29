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
        ('a', (SubProofKind.existential, {('FORALL', 'x', ('P', 'x'))})),
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
                ((10, 'CONTR'), dict(), set(), set(), set()),
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
                ((10, 'CONTR', ('P',), 5), set(), set(), {'y'}),
                ((10, 'x', ('P', 'x'), 5), set(), set(), {'x', 'y'}),
                ((10, 'x', ('Q', 'x'), 5), {'P', 'Q'}, set(), {'y'}),
                ((10, 'f', ('P', 'f'), 5), {'P'}, {'f'}, {'y'}),
            ])
        def test_bad(proof, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, {5: ('EXISTS', 'y', ('P', 'y'))},
                               seen_predicates, seen_functions, seen_objects)

    @staticmethod
    @pytest.fixture
    def seen_predicates():
        return {'_Pred_P', '_Pred_Q'}

    @staticmethod
    @pytest.fixture
    def seen_functions():
        return {'_Func_f', '_Func_g'}

    @staticmethod
    @pytest.fixture
    def seen_objects():
        return {'_Obj_a', '_Obj_b', '_Obj_x', '_Obj_y'}

    class TestConjunctionIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
                ((50, ('AND', ('P',), ('Q',)), [40, 30], 'CI'),
                 {30: ('P',), 40: ('Q',)}),
                ((50, ('AND', ('P',), ('P',)), [30], 'CI'),
                 {30: ('P',)}),
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
            'proof, facts_by_index', [
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30], 'CI'),
                 {30: ('P', ('f', 'x'))}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [40], 'CI'),
                 {40: ('Q', 'y')}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [30, 40, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
                ((50, ('AND', ('P', ('f', 'x')), ('Q',)), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
                ((50, ('AND', ('P',), ('Q', 'y')), [30, 40], 'CI'),
                 {30: ('P', ('f', 'x')), 40: ('Q', 'y')}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates, seen_functions, seen_objects)

    class TestConjunctionElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('P',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))}),
                ((50, ('Q',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))}),
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
            'proof, facts_by_index', [
                ((50, ('P',), [], 'CE'), dict()),
                ((50, ('P',), [40, 40], 'CE'), {40: ('AND', ('P',), ('Q',))}),
                ((50, ('R',), [40], 'CE'), {40: ('AND', ('P',), ('Q',))}),
                ((50, ('P',), [40], 'CE'), {40: ('OR', ('P',), ('Q',))}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates, seen_functions, seen_objects)

    class TestDisjunctionIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('Q', 'y')}),
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
            'proof, facts_by_index', [
                ((50, ('AND', ('P', ('f', 'x')), ('Q', 'y')), [40], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [40, 30], 'DI'),
                 {40: ('P', ('f', 'x')), 30: ('Q', 'y')}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q', 'y')), [30], 'DI'),
                 {40: ('P', ('f', 'x'))}),
                ((50, ('OR', ('P', ('f', 'x')), ('Q',)), [40], 'DI'),
                 {40: ('Q', 'y')}),
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
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates,
                                                       seen_functions,
                                                       seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

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
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index, seen_predicates,
                               seen_functions, seen_objects)

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
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates,
                                                       seen_functions,
                                                       seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

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
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestImplicationElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('Q',), [20, 30], 'IE'),
                 {20: ('IMPLIES', ('P',), ('Q',)),
                  30: ('P',)}),
                ((50, ('Q',), [30, 20], 'IE'),
                 {20: ('IMPLIES', ('P', 'x'), ('Q',)),
                  30: ('P', 'x')}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('Q',), [20, 30, 30], 'IE'),
                 {20: ('IMPLIES', ('P',), ('Q',)),
                  30: ('P',)}),

                ((50, ('Q',), [20], 'IE'),
                 {20: ('IMPLIES', ('P',), ('Q',)),
                  30: ('P',)}),

                ((50, ('Q',), [20, 30], 'IE'),
                 {20: ('IMPLIES', ('P',), ('Q', 'x')),
                  30: ('P',)}),

                ((50, ('Q',), [20, 30], 'IE'),
                 {20: ('IMPLIES', ('P',), ('Q',)),
                  30: ('P', 'x')}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestNegationIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('NOT', ('P', 'x')), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {'CONTR'})}),
                ((50, ('NOT', ('P',)), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P',)}, {'CONTR', ('Q',)})}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('NOT', ('P', 'x')), [], 'NI'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {'CONTR'})}),
                ((50, ('NOT', ('P', 'x')), [20, 20], 'NI'),
                 {20: (SubProofKind.conditional, {('P', 'x')}, {'CONTR'})}),
                ((50, ('NOT', ('P', 'x')), [20], 'NI'),
                 {20: (SubProofKind.universal, {('P', 'x')}, {'CONTR'})}),
                ((50, (('P',), ('Q',)), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P',)}, {'CONTR'})}),
                ((50, ('NOT', ('P', 'x')), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P', 'x'), ('Q',)}, {'CONTR'})}),
                ((50, ('NOT', ('P', 'x')), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P', 'y')}, {'CONTR'})}),
                ((50, ('NOT', ('P',)), [20], 'NI'),
                 {20: (SubProofKind.conditional, {('P',)}, {('Q',)})}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestNegationElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('P', 'x'), [20], 'NE'),
                 {20: ('NOT', ('NOT', ('P', 'x')))}),
                ((50, ('NOT', ('P', 'x')), [20], 'NE'),
                 {20: ('NOT', ('NOT', ('NOT', ('P', 'x'))))}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('P', 'x'), [20], 'NE'),
                 {20: ('NOT', ('P', 'x'))}),
                ((50, ('NOT', ('P', 'x')), [30], 'NE'),
                 {20: ('NOT', ('NOT', ('NOT', ('P', 'x'))))}),
                ((50, ('NOT', ('P', 'x')), [20], 'NE'),
                 {20: ('NOT', ('Q', ('NOT', ('P', 'x'))))}),
                ((50, ('Q', ('P', 'x')), [20], 'NE'),
                 {20: ('NOT', ('NOT', ('NOT', ('P', 'x'))))}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestContradictionIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [20, 30], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, 'CONTR', [30, 20], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, 'CONTR', [30], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, 'CONTR', [20], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, 'CONTR', [30, 20, 30], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, ('P', 'x'), [30, 20], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, ('NOT', ('P', 'x')), [30, 20], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('P', 'x')}),
                ((50, 'CONTR', [30, 20], 'XI'),
                 {20: ('NOT', ('P', 'x')),
                  30: ('NOT', ('P', 'x'))}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestContradictionElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [20], 'XE'),
                 {20: 'CONTR'}),
                ((50, ('_Pred_P', '_Obj_x'), [20], 'XE'),
                 {20: 'CONTR'}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [], 'XE'),
                 {20: 'CONTR'}),
                ((50, 'CONTR', [20, 20], 'XE'),
                 {20: 'CONTR'}),
                ((50, ('Unknown_Property',), [20], 'XE'),
                 {20: 'CONTR'}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    class TestUniversalIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, objects', [
                ((50, ('FORALL', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'x'}),
                ((50, ('FORALL', 'y', ('Q',)), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z'), ('Q',)})},
                 {'x'}),
                ((50, ('FORALL', 'y', ('P', ('f', 'y'), 'y', 'z')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'x', {('P', ('f', 'x'), 'x', 'z')})},
                 {'x', 'z'}),
            ])
        def test_good(proof, facts_by_index, objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index, set(), set(), objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert set() == preds
            assert set() == funcs
            assert objects | {'y'} == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, predicates, functions, objects', [
                ((50, ('FORALL', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', set())},
                 {'x'}, set(), set()),
                ((50, ('EXISTS', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'P'}, set(), {'z'}),
                ((50, ('FORALL', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'P', 'y'}, set(), {'z'}),
                ((50, ('FORALL', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'P'}, {'y'}, {'z'}),
                ((50, ('FORALL', 'y', ('Q', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'P', 'Q'}, set(), {'z'}),
                ((50, ('FORALL', 'y', ('P', 'y')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'z', {('P', 'z')})},
                 {'P'}, set(), {'y', 'z'}),
                ((50, ('FORALL', 'y', ('P', ('g', 'y'), 'y', 'z')), [10], 'AI'),
                 {10: (SubProofKind.universal, 'x', {('P', ('f', 'x'), 'x', 'z')})},
                 {'P'}, {'f'}, {'x', 'z'}),
            ])
        def test_bad(proof, facts_by_index, predicates, functions, objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               predicates, functions, objects)

    class TestUniversalElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, objects', [
                ((50, ('P', 'x'), [10], 'AE'),
                 {10: ('FORALL', 'y', ('P', 'y'))},
                 {'x'}),
                ((50, ('P', ('f', 'a'), 'a', 'z'), [10], 'AE'),
                 {10: ('FORALL', 'y', ('P', ('f', 'y'), 'y', 'z'))},
                 {'x', 'a'}),
            ])
        def test_good(proof, facts_by_index, objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index, set(), set(), objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert set() == preds
            assert set() == funcs
            assert objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, predicates, functions', [
                ((50, ('P', 'x'), [10], 'AE'),
                 {10: ('EXISTS', 'y', ('P', 'y'))},
                 {'x'}, set()),
                ((50, ('P', 'x'), [10, 10], 'AE'),
                 {10: ('FORALL', 'y', ('P', 'y'))},
                 {'x'}, set()),
                ((50, ('P', ('f', 'y'), 'y'), [10], 'AE'),
                 {10: ('FORALL', 'g', ('P', 'g', 'y'))}, set(), set()),
                ((50, ('P', ('f', 'y'), 'y'), [10], 'AE'),
                 {10: ('FORALL', 'x', ('P', 'x', 'x'))}, set(), set()),
                ((50, ('P', ('f', 'y'), 'y'), [10], 'AE'),
                 {10: ('FORALL', 'x', ('P', ('f', 'x'), 'x'))}, {'y'}, set()),
                ((50, ('P', ('f', 'y'), 'y'), [10], 'AE'),
                 {10: ('FORALL', 'x', ('P', ('f', 'x'), 'x'))}, set(), {'y'}),
                ((50, ('P', ('f', 'y'), 'y'), [10], 'AE'),
                 {10: ('FORALL', 'x', ('P', ('f', 'x'), 'x'))}, set(), set()),
            ])
        def test_bad(proof, facts_by_index, predicates, functions):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               predicates, functions, set())

    class TestExistenceIntroduction:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('EXISTS', 'x', ('P', ('f', 'x'), 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs - {'x'}

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, predicates, functions, objects', [
                ((50, ('FORALL', 'x', ('P', ('f', 'x'), 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), set(), set()),
                ((50, ('EXISTS', 'g', ('P', 'g', 'y')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), set(), set()),
                ((50, ('EXISTS', 'x', ('P', ('f', 'x'), 'x')), [10, 10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), set(), set()),
                ((50, ('EXISTS', 'x', ('P', 'x', 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), set(), set()),
                ((50, ('EXISTS', 'x', ('P', ('f', 'x'), 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, {'x'}, set(), set()),
                ((50, ('EXISTS', 'x', ('P', ('f', 'x'), 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), {'x'}, set()),
                ((50, ('EXISTS', 'x', ('P', ('f', 'x'), 'x')), [10], 'EI'),
                 {10: ('P', ('f', 'y'), 'y')}, set(), set(), {'x'}),
            ])
        def test_bad(proof, facts_by_index, predicates, functions, objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               predicates, functions, objects)

    class TestExistenceElimination:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('Q', 'z'), [10], 'EE'),
                 {10: (SubProofKind.existential, {('Q', 'z')})}),
                ((50, ('Q',), [10], 'EE'),
                 {10: (SubProofKind.existential, {('P', 'z'), ('Q',)})}),
                ((50, ('P', 'z'), [10], 'EE'),
                 {10: (SubProofKind.existential, {('P', 'z'), ('Q',)})}),
            ])
        def test_good(proof, facts_by_index):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index, set(), set(), set())
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert set() == preds
            assert set() == funcs
            assert set() == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, ('P', 'z'), [10], 'EE'),
                 {10: (SubProofKind.existential, {('Q', 'z')})}),
                ((50, ('P',), [10], 'EE'),
                 {10: (SubProofKind.existential, {('P', 'z'), ('Q',)})}),
                ((50, ('P', 'z'), [], 'EE'),
                 {10: (SubProofKind.existential, {('P', 'z'), ('Q',)})}),
                ((50, ('P', 'z'), [10, 10], 'EE'),
                 {10: (SubProofKind.existential, {('P', 'z'), ('Q',)})}),
            ])
        def test_bad(proof, facts_by_index):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               set(), set(), set())

    class TestSupposition:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [], 'S'), dict()),
                ((50, ('_Pred_P', '_Obj_x'), [], 'S'), {20: ('Q',)}),
                ((50, ('P', 'x'), [], 'S'), {20: ('Q',)}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds - {'P'}
            assert seen_functions == funcs
            assert seen_objects == objs - {'x'}

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index, predicates, functions, objects', [
                ((50, ('CONTR',), [], 'S'), dict(), set(), set(), set()),
                ((50, ('P', ('CONTR', 'x')), [], 'S'), dict(), set(), set(), set()),
                ((50, ('P',), [], 'S'), {50: ('P',)}, set(), set(), set()),
                ((50, ('P',), [], 'S'), {20: ('P',)}, set(), set(), set()),
                ((50, ('Q',), [20], 'S'), {20: ('Q',)}, set(), set(), set()),
                ((50, ('Q',), [], 'S'), {20: ('Q',)}, set(), set(), set()),
                ((50, ('P',), [], 'S'), dict(), set(), {'P'}, set()),
                ((50, ('P',), [], 'S'), dict(), set(), set(), {'P'}),
                ((50, ('P', ('f', 'x')), [], 'S'), dict(), {'f'}, set(), set()),
                ((50, ('P', ('f', 'x')), [], 'S'), dict(), set(), set(), {'f'}),
                ((50, ('P', 'x'), [], 'S'), dict(), {'x'}, set(), set()),
                ((50, ('P', 'x'), [], 'S'), dict(), set(), {'x'}, set()),
            ])
        def test_bad(proof, facts_by_index, predicates, functions, objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               predicates, functions, objects)

    class TestReiteration:

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [20], 'RE'),
                 {20: 'CONTR'}),
                ((50, ('P', 'x'), [20], 'RE'),
                 {20: ('P', 'x')}),
            ])
        def test_good(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            index, predicate, *_ = proof
            facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                       seen_predicates, seen_functions, seen_objects)
            assert facts_by_index == {k: v for k, v in facts.items() if k != index}
            assert facts[index] == predicate
            assert seen_predicates == preds
            assert seen_functions == funcs
            assert seen_objects == objs

        @staticmethod
        @pytest.mark.parametrize(
            'proof, facts_by_index', [
                ((50, 'CONTR', [], 'RE'),
                 {20: 'CONTR'}),
                ((50, 'CONTR', [20, 20], 'RE'),
                 {20: 'CONTR'}),
                ((50, ('Q', 'x'), [20], 'RE'),
                 {20: ('P', 'x')}),
            ])
        def test_bad(proof, facts_by_index, seen_predicates, seen_functions, seen_objects):
            with pytest.raises(InvalidProof):
                validate_proof(proof, facts_by_index,
                               seen_predicates, seen_functions, seen_objects)

    @staticmethod
    @pytest.mark.parametrize(
        'proof, facts_by_index, result_facts', [
            ((50, []),
             {20: 'CONTR'},
             {20: 'CONTR', 50: (SubProofKind.arbitrary, set())}),
            ((30, [
                (40, ('P',), [10], 'RE'),
                (50, ('Q',), [20], 'RE')
            ]),
             {10: ('P',), 20: ('Q',)},
             {10: ('P',), 20: ('Q',), 30: (SubProofKind.arbitrary, {('P',), ('Q',)})}),
            ((30, [
                (40, [
                    (50, ('P',), [10], 'RE'),
                ]),
                (60, [
                    (70, ('Q',), [20], 'RE'),
                ]),
            ]),
             {10: ('P',), 20: ('Q',)},
             {10: ('P',), 20: ('Q',), 30: (SubProofKind.arbitrary, {('P',), ('Q',)})}),
        ])
    def test_arbitrary_sub_proof(proof, facts_by_index, result_facts,
                                 seen_predicates, seen_functions, seen_objects):
        facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                   seen_predicates, seen_functions, seen_objects)
        assert result_facts == facts
        assert seen_predicates == preds
        assert seen_functions == funcs
        assert seen_objects == objs

    @staticmethod
    @pytest.mark.parametrize(
        'proof, facts_by_index, result_facts', [
            ((30, [
                (40, ('P',), [], 'S'),
                (50, ('Q',), [20], 'RE'),
                (60, ('R',), [], 'S'),
                (70, ('Q',), [50], 'RE'),
            ]),
             {20: ('Q',)},
             {20: ('Q',), 30: (SubProofKind.conditional,
                               {('P',), ('R',)},
                               {('P',), ('Q',), ('R',)})}),
            ((30, [
                (35, ('R',), [], 'S'),
                (40, [
                    (50, ('P',), [10], 'RE'),
                ]),
                (60, [
                    (70, ('S',), [], 'S'),
                ]),
                (80, ('Q',), [20], 'RE'),
            ]),
             {10: ('P',), 20: ('Q',)},
             {10: ('P',), 20: ('Q',), 30: (SubProofKind.conditional,
                                           {('R',)},
                                           {('R',), ('Q',), ('P',)})}),
        ])
    def test_conditional_sub_proof(proof, facts_by_index, result_facts,
                                   seen_predicates, seen_functions, seen_objects):
        facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                   seen_predicates, seen_functions, seen_objects)
        assert result_facts == facts
        assert seen_predicates == preds - {'R', 'Q', 'P', 'S'}
        assert seen_functions == funcs
        assert seen_objects == objs

    @staticmethod
    @pytest.mark.parametrize(
        'proof, facts_by_index, result_facts', [
            ((35, [(40, 'x'), ]),
             dict(),
             {35: (SubProofKind.universal, 'x', set())}),
            ((35, [
                (40, 'x'),
                (50, ('P', 'x'), [20], 'AE'),
                (60, ('Q', 'x'), [30], 'AE'),
                (70, ('R',), [10], 'RE'),
            ]),
             {10: ('R',),
              20: ('FORALL', 'x', ('P', 'x')),
              30: ('FORALL', 'y', ('Q', 'y'))},
             {10: ('R',),
              20: ('FORALL', 'x', ('P', 'x')),
              30: ('FORALL', 'y', ('Q', 'y')),
              35: (SubProofKind.universal, 'x', {('P', 'x'), ('Q', 'x'), ('R',)})}),
            ((35, [
                (40, 'x'),
                (45, [
                    (50, 'y'),
                    (55, ('R',), [10], 'RE'),
                ]),
                (47, [
                    (50, 'z', ('Q', 'z'), 33),
                    (55, ('R',), [10], 'RE'),
                ]),
                (60, ('Q', 'x'), [30], 'AE'),
                (65, [
                    (80, ('T',), [], 'S'),
                ]),
                (90, [
                    (100, ('P',), [20], 'RE'),
                ]),
            ]),
             {10: ('R',),
              20: ('P',),
              33: ('EXISTS', 'y', ('Q', 'y')),
              30: ('FORALL', 'y', ('Q', 'y'))},
             {10: ('R',),
              20: ('P',),
              30: ('FORALL', 'y', ('Q', 'y')),
              33: ('EXISTS', 'y', ('Q', 'y')),
              35: (SubProofKind.universal, 'x', {('Q', 'x'), ('P',)})}),
        ])
    def test_universal_sub_proof(proof, facts_by_index, result_facts,
                                 seen_predicates, seen_functions, seen_objects):
        facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                   seen_predicates, seen_functions, seen_objects)
        assert result_facts == facts
        # assert seen_predicates == preds - {'R', 'Q', 'P', 'S'}
        # assert seen_functions == funcs
        # assert seen_objects == objs

    @staticmethod
    @pytest.mark.parametrize(
        'proof, facts_by_index, result_facts', [
            ((35, [
                (40, 'x', ('P', 'x'), 30),
                (50, ('Q', 'z'), [20], 'RE'),
            ]),
             {20: ('Q', 'z'),
              30: ('EXISTS', 'y', ('P', 'y'))},
             {20: ('Q', 'z'),
              30: ('EXISTS', 'y', ('P', 'y')),
              35: (SubProofKind.existential, {('Q', 'z')})}),
            ((35, [
                (40, 'x', ('P', 'x'), 20),
                (45, [
                    (50, 'y'),
                    (55, ('R',), [10], 'RE'),
                ]),
                (47, [
                    (50, 'z', ('Q', 'z'), 33),
                    (55, ('R',), [10], 'RE'),
                ]),
                (60, ('P', 'x'), [40], 'RE'),
                (63, ('Q',), [30], 'AE'),
                (65, [
                    (80, ('U',), [], 'S'),
                ]),
                (90, [
                    (100, ('T',), [15], 'RE'),
                ]),
            ]),
             {10: ('R',),
              15: ('T',),
              20: ('EXISTS', 'x', ('P', 'x')),
              30: ('FORALL', 'y', ('Q',)),
              33: ('EXISTS', 'y', ('Q', 'y'))},
             {10: ('R',),
              15: ('T',),
              20: ('EXISTS', 'x', ('P', 'x')),
              30: ('FORALL', 'y', ('Q',)),
              33: ('EXISTS', 'y', ('Q', 'y')),
              35: (SubProofKind.existential, {('Q',), ('T',)})}),
        ])
    def test_existential_sub_proof(proof, facts_by_index, result_facts,
                                   seen_predicates, seen_functions, seen_objects):
        facts, preds, funcs, objs = validate_proof(proof, facts_by_index,
                                                   seen_predicates, seen_functions, seen_objects)
        assert result_facts == facts
        # assert seen_predicates == preds - {'R', 'Q', 'P', 'S'}
        # assert seen_functions == funcs
        # assert seen_objects == objs


class TestVerifyProof:

    @staticmethod
    @pytest.mark.parametrize('proof', [
        """
        (SUBP 5
            (10 (AND (NOT (P a)) (NOT (P b))) ([] S))
            (SUBP 15
                (20 (OR (P a) (P b)) ([] S))
                (SUBP 25
                    (30 (P a) ([] S))
                    (40 (NOT (P a)) ([10] CE))
                    (50 (CONTR) ([30, 40] XI)))
                (SUBP 55
                    (60 (P b) ([] S))
                    (70 (NOT (P b)) ([10] CE))
                    (80 (CONTR) ([60, 70] XI)))
                (90 (CONTR) ([20, 25, 55] DE)))
            (100 (NOT (OR (P a) (P b))) ([15] NI)))
        """,
        """
        (SUBP 5
            (10 (AND (NOT (P)) (NOT (P b))) ([] S))
            (SUBP 15
                (20 (OR (P) (P b)) ([] S))
                (SUBP 25
                    (30 (P) ([] S))
                    (40 (NOT (P)) ([10] CE))
                    (50 (CONTR) ([40, 30] XI)))
                (SUBP 55
                    (60 (P b) ([] S))
                    (70 (NOT (P b)) ([10] CE))
                    (80 (CONTR) ([70, 60] XI)))
                (90 (CONTR) ([25, 20, 55] DE)))
            (100 (NOT (OR (P) (P b))) ([15] NI)))
        """,
    ])
    def test_good(proof):
        assert verifyProof(proof) == 'V'

    @staticmethod
    @pytest.mark.parametrize('proof', [
        """
        (SUBP 5
            (10 (AND (NOT (P a)) (NOT (P b))) ([] S))
            (SUBP 10
                (20 (OR (P a) (P b)) ([] S))
                (SUBP 25
                    (30 (P a) ([] S))
                    (40 (NOT (P a)) ([10] CE))
                    (50 (CONTR) ([30, 40] XI)))
                (SUBP 55
                    (60 (P b) ([] S))
                    (70 (NOT (P b)) ([10] CE))
                    (80 (CONTR) ([60, 70] XI)))
                (90 (CONTR) ([20, 25, 55] DE)))
            (100 (NOT (OR (P a) (P b))) ([15] NI)))
        """,
        """
        (SUBP 5
            (10 (AND (NOT (P a)) (NOT (P b))) ([] S))
            (SUBP 15
                (20 (OR (P a) (P b)) ([] S))
                (SUBP 25
                    (30 (P a) ([] S))
                    (35 (A P) ([] S))
                    (40 (NOT (P a)) ([10] CE))
                    (50 (CONTR) ([30, 40] XI)))
                (SUBP 55
                    (60 (P b) ([] S))
                    (70 (NOT (P b)) ([10] CE))
                    (80 (CONTR) ([60, 70] XI)))
                (90 (CONTR) ([20, 25, 55] DE)))
            (100 (NOT (OR (P a) (P b))) ([15] NI)))
        """,
        """
        (SUBP 5
            (10 (AND (NOT (P a)) (NOT (P b))) ([] S))
            (SUBP 15
                (20 (OR (P a) (P b)) ([] S))
                (SUBP 25
                    (30 (P a) ([] S))
                    (40 (NOT (P a)) ([10] CE))
                    (50 (CONTR) ([30, 40] XI)))
                (SUBP 55
                    (60 (P b) ([] S))
                    (70 (NOT (P b)) ([10] CE))
                    (80 (CONTR) ([60, 70] XI)))
                (90 (CONTR) ([25, 55] DE)))
            (100 (NOT (OR (P a) (P b))) ([15] NI)))
        """,
    ])
    def test_bad(proof):
        assert verifyProof(proof) == 'I'
