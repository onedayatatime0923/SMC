; benchmark generated as a simple test case
(set-info :status sat)
(declare-fun y__0 () Real)
(declare-fun y__1 () Real)
(assert (= y__0 2.0))
(assert (= y__1 (+ y__0 3.0)))
(assert (> y__1 6.0))
(check-sat)
