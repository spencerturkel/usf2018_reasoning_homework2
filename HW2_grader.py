"""This is basically the file I will use to grade your HW2 submissions.
Obviously, the test problems hard-coded in here will be different. You
can use this to test your code. I'm using Python 3.4+.
- Dr. Licato
"""

import random
import traceback
import time

studentName = "TestStudent"
# there will likely be 5-10 problems
problems = [
    # note that the nice indentation shown here is not required for well-formedness. It's just for your reading convenience.
    """
    (SUBP 5
  (10 (FORALL x (IMPLIES (P x) (Q x))) ([] S))
  (20 (FORALL x (P x)) ([] S))
  (SUBP 25
    (30 (UCONST a) ([] UCONST))
    (40 (P a) ([20] AE))
    (50 (IMPLIES (P a) (Q a)) ([10] AE))
    (60 (Q a) ([40,50] IE))
  )
  (70 (FORALL x (Q x)) ([25] AI))
)
    """,
    """
    (SUBP 5
  (10 (EXISTS x (P x)) ([] S))
  (20 (FORALL x (IMPLIES (P x) (Q a))) ([] S))
  (SUBP 25
    (30 (ECONST b (P b)) ([10] ECONST))
    (40 (IMPLIES (P b) (Q a)) ([20] AE))
    (50 (Q a) ([30,40] IE))
  )
  (60 (Q a) ([25] EE))
)
    """,
    """
    (SUBP 5
        (10 (AND (NOT (P a)) (NOT (P b))) ([] S))
        (SUBP 15
            (20 (OR (P a) (P b)) ([] S))
            (SUBP 25
                  (30 (P a) ([] S))
                  (40 (NOT (P a)) ([10] CE))
                  (50 (CONTR) ([30,40] XI))
            )
            (SUBP 55
                (60 (P b) ([] S))
                (70 (NOT (P b)) ([10] CE))
                (80 (CONTR) ([60,70] XI))
            )
            (90 (CONTR) ([20,25,55] DE))
        )
        (100 (NOT (OR (P a) (P b))) ([15] NI))
    )
    """,
    """	
	(SUBP 5
		(10 (EXISTS x (P x)) ([] S))
		(20 (FORALL x (IMPLIES (P x) (Q a))) ([] S))
		(SUBP 25
			(30 (ECONST b (P b)) ([10] ECONST))
			(40 (IMPLIES (P b) (Q a)) ([20] AE))
			(50 (Q a) ([30,50] IE))
		)
		(60 (Q a) ([25] EE))
	)""",
    """
	(SUBP 5
		(10 (FORALL x (IMPLIES (P x) (Q x))) ([] S))
		(20 (P a) ([] S))
		(30 (IMPLIES (P a) (Q a)) ([10] AE))
		(40 (Q a) ([20,30] IE))
	)""",
    """
	(SUBP 5
		(10 (FORALL x (IMPLIES (P x) (Q x))) ([] S))
		(20 (P a) ([] S))
		(30 (IMPLIES (P a) (Q a)) ([10] AE))
		(40 (P a) ([20,30] IE))
	)""",
    """
	(SUBP 5
		(10 (FORALL x (IMPLIES (P x) (Q x))) ([] S))
		(20 (FORALL x (P x)) ([] S))
		(SUBP 25
			(30 (UCONST a) ([] UCONST))
			(40 (P a) ([20] AE))
			(50 (IMPLIES (P a) (Q a)) ([10] AE))
			(60 (Q a) ([40,50] IE))
		)
		(70 (FORALL x (Q x)) ([25] AI))
	)"""
]
answers = ['V', 'V', 'V', 'V', 'V', 'I', 'V']

maxProblemTimeout = 30

outFile = open("grade_" + studentName + ".txt", 'w')


def prnt(S):
    global outFile
    outFile.write(str(S) + "\n")
    print(S)


try:
    F = open("hw2.py", 'r', encoding="utf-8")
    exec("".join(F.readlines()))
except Exception as e:
    prnt("Couldn't open or execute 'hw2.py': " + str(traceback.format_exc()))
    prnt("FINAL SCORE: 0")
    exit()

currentScore = 100
for i in range(len(problems)):
    P = problems[i]
    A = answers[i]

    prnt('=' * 30)
    prnt("TESTING ON INPUT PROBLEM:")
    prnt(P)
    prnt("CORRECT OUTPUT:")
    prnt(str(A))
    prnt("YOUR OUTPUT:")
    try:
        startTime = time.time()
        result = verifyProof(P)
        prnt(result)
        endTime = time.time()
        if endTime - startTime > maxProblemTimeout:
            prnt("Time to execute was " + str(int(endTime - startTime)) + " seconds; this is too long (-10 points)")
        elif result == A:
            prnt("Correct!")
        else:
            prnt("Incorrect (-10 points)")
            currentScore -= 10

    except Exception as e:
        prnt("Error while executing this problem: " + str(traceback.format_exc()))
        currentScore -= 10

prnt('=' * 30)
prnt('=' * 30)
prnt('=' * 30)
prnt("FINAL SCORE:" + str(currentScore))
