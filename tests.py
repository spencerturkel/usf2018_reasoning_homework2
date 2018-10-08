from hw2 import *

import pytest


class TestLexer:
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
    def test_listing(self, string, result):
        assert list(Lexer(string)) == result

    def test_peeking_and_nexting(self):
        lexer = Lexer(' [\t] \t  0   Ab123')
        for token in ['[', ']', 0, 'Ab123']:
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
    l = ListLexer([1, 2, 3])
    assert 1 == l.peek()
    assert next(l) == 1
    assert 2 == l.peek()


@pytest.mark.parametrize('ast,lexer', [
    ((10, []), ListLexer(['(', 'SUBP', 10, ')'])),
    ((10, 'x'), ListLexer(['(', 10,
                           '(', 'UCONST', 'x', ')',
                           '(', '[', ']', 'UCONST', ')', ')'])),
    ((10, 'x', ('P', []), 5), ListLexer(['(', 10,
                                         '(', 'ECONST', 'x', '(', 'P', ')', ')',
                                         '(', '[', 5, ']', 'ECONST', ')', ')'])),
])
def test_parse(ast, lexer):
    assert ast == parse(lexer)
