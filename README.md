![Build Status](https://travis-ci.com/spencerturkel/usf2018_reasoning_homework2.svg?token=gm1zuwtz6yWqd9Rwapxf&amp;branch=master)

# What is this?
This project was the second homework for the Computationally Modelling Reasoning course at the University of South Florida.

Our assignment is to verify [Fitch-style proofs](https://en.wikipedia.org/wiki/Fitch_notation) of formulae in [First-Order Logic](https://en.wikipedia.org/wiki/First-order_logic).

# Requirements
- Write a Python 3.4+ module `hw2.py` which contains a function `verifyProof(P)`.
- `hw2.py` must not contain any outputting code (such as `print` calls).
- The input `P` to `verifyProof(P)` will be an s-expression of a Fitch proof.
    * [Lexemes](#lexemes)
        * There may be arbitrary whitespace between lexemes
    * [Grammar](#grammar)
    * [Inference Rules](#inference-rules)
- `verifyProof(P)` must return:
    * `'V'` if `P` is a valid proof
    * `'I'` if `P` is an invalid proof
- `verifyProof(P)` may assume that:
    * All input is well-formed.
    * All operators (except `NOT`) have two arguments.
    * All symbols will be lower-case alphanumeric strings.
# Input
## Lexemes
```
(
)
[
]
,
SUBP
quantified_constant = UCONST | ECONST
op = FORALL
   | EXISTS
   | CONTR
   | AND
   | OR
   | IMPLIES
   | NOT
inference-rule = S
               | CI
               | CE
               | DI
               | DE
               | II
               | IE
               | NI
               | NE
               | AI
               | AE
               | EI
               | EE
               | XI
               | XE
               | RE
index = \d+
symbol = [A-Za-z0-9]+
```
## Grammar
Let `ε` be the empty string.

```
proof = ( line )
line = SUBP number many-proofs
     | index predicate justification
     | index universal-constant
     | index existential-constant
many-proofs = proof many-proofs | ε
predicate = ( expr )
justification = ( [ indices ] inference-rule )
universal-constant = ( UCONST symbol ) ( [ ] UCONST )
existential-constant = ( ECONST symbol predicate ) ( [ index ] ECONST )
expr = FORALL symbol predicate
     | EXISTS symbol predicate
     | AND argument argument
     | OR argument argument
     | IMPLIES argument argument
     | NOT argument
     | CONTR
     | symbol many-arguments
indices = index trailing-indices | ε
many-arguments = argument many-arguments | ε
trailing-indices = , index trailing-indices | ε
argument = symbol | predicate
```
# Inference Rules
`R |- {P_0, P_1, ..., P_n} -> Q` means that inference rule `R` justifies `Q` within the context `{P_i | 0 <= i < n}`.
`C => p` symbolizes a sub-proof of `p` given the additional context `C`.

## Formal Rules
```
CI |- {p, q} -> (AND p q)
CE |- {(AND p q)} -> p
CE |- {(AND p q)} -> q
DI |- {p} -> (OR p q)
DI |- {q} -> (OR p q)
DE |- {{p} => r, {q} => r, (OR p q)} -> r
II |- {{p} => q} -> (IMPLIES p q)
IE |- {(IMPLIES p q), p} -> q
NI |- {{p} => (CONTR)} -> (NOT p)
NE |- {(NOT (NOT p))} -> p
XI |- {p, (NOT p)} -> (CONTR)
XE |- {(CONTR)} -> p
AI |- {{x} => P x} -> (FORALL y (P y))
AE |- {(FORALL x (P x)), y} -> P y
EI |- {P x} -> (EXISTS y (P y))
EE |- {(EXISTS x (P x)), {y, P y} => q} -> q
```

## Informal Rules
```
S |- {} -> p
```
This is the *supposition*, which justifies anything we assume at the start of the proof.

```
UCONST |- {} -> p
```
This is the *universal constant*, which justifies an arbitrary object when proving a universal proposition.

```
ECONST |- {(EXISTS x (P x))} -> (ECONST y (P y))
```
This is the *existential constant*, which justifies:
- the existence of an object `y`
- and the proof of its property `P`

when using an existential proposition.
