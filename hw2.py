import random


def verifyProof(P):
    """F: A string which is an S-expression of a well-formed Fitch-style
    proof.
    Returns either a string or integer:
    “E” – If P was not well-formed (GRADUATE STUDENTS ONLY)
    “I” – If P was well-formed, but not a valid proof,
    “V” – If P was well-formed, and a valid proof.
    """
    to_return = random.choice(["E", "I", "V"])
    if to_return == "i":
        return random.randint(0, 100)
    else:
        return to_return
