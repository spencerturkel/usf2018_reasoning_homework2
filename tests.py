from hw2 import *

import pytest


class TestLexer:
    @pytest.mark.parametrize(['string', 'result'], [
        ('UCONST ECONST', set(QuantifiedConstant)),
        ('FORALL EXISTS CONTR AND OR IMPLIES NOT', set(Op)),
        ('S CI CE DI DE II IE NI NE AI AE EI EE XI XE RE', set(InferenceRule)),
        ('()[],SUBP', set(CommonToken)),
        ('AND ORr', {Op.conjunction, 'ORr'}),
        ('ANDpq OR', {'ANDpq', Op.disjunction}),
    ])
    def test_set(self, string, result):
        assert set(Lexer(string)) == result

    @pytest.mark.parametrize(['string', 'result'], [
        ('az,() 123', ['az', CommonToken.comma, CommonToken.left_parenthesis,
                       CommonToken.right_parenthesis, 123]),
        ('(10 (AND (x) (y)) ([] S))', [CommonToken.left_parenthesis, 10, CommonToken.left_parenthesis,
                                       Op.conjunction, CommonToken.left_parenthesis, 'x', CommonToken.right_parenthesis,
                                       CommonToken.left_parenthesis, 'y', CommonToken.right_parenthesis,
                                       CommonToken.right_parenthesis, CommonToken.left_parenthesis,
                                       CommonToken.left_bracket, CommonToken.right_bracket,
                                       InferenceRule.supposition, CommonToken.right_parenthesis,
                                       CommonToken.right_parenthesis]),
        (' [\t] \n\t  0   ab123', [CommonToken.left_bracket, CommonToken.right_bracket,
                                   0, 'ab123'])
    ])
    def test_listing(self, string, result):
        assert list(Lexer(string)) == result

    def test_peeking_and_nexting(self):
        lexer = Lexer(' [\t] \t  0   Ab123')
        for token in [CommonToken.left_bracket, CommonToken.right_bracket, 0, 'Ab123']:
            assert lexer.peek() == token
            assert next(lexer) == token
        assert lexer.peek() is None
        with pytest.raises(StopIteration):
            next(lexer)


class ListLexer:
    """
    An iterable returning values from a list, and allowing a .peek() operation.
    """

    def __init__(self, values):
        self.values = values

    def __next__(self):
        try:
            return self.values.pop(0)
        except IndexError:
            raise StopIteration

    def peek(self):
        return self.values[0] if len(self.values) > 0 else None


def test_list_lexer():
    l = ListLexer([1, 2, 3])
    assert 1 == l.peek()
    assert next(l) == 1
    assert 2 == l.peek()
