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
    * All object and predicate symbols will be lower-case alphanumeric strings.
# Input
## Lexemes
```
(
)
[
]
,
SUBP
op = FORALL
   | EXISTS
   | UCONST
   | EXCONST
   | CONTR
   | AND
   | OR
   | IMPLIES
   | NOT
index = [0-9]+
object = [a-z0-9]+
predicate = {w | w in Regex("[A-Z][a-z0-9]+") && w not in op}
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
```
## Grammar
```
proof = ( line )
line = SUBP number many-proofs | index expr justification
many-proofs = proof many-proofs |
expr = object | list
justification = ( [ indices ] inference-rule )
list = ( formula )
indices = index , indices |
formula = FORALL symbol expr
        | EXISTS symbol expr
        | UCONST symbol
        | ECONST symbol expr
        | AND expr expr
        | OR expr expr
        | IMPLIES expr expr
        | NOT expr
        | CONTR
        | predicate
```
# Inference Rules
```

```
