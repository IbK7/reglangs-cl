from problems.regular import FiniteAutomaton


class EvenPairs(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 5, [0,1,3], {(0,1): "a", (1,1): "a", (1,2): "b", (2,1): "a", (2,2): "b", (0,3): "b", (3,3): "b", (3,4): "a", (4,3): "b", (4,4): "a"})


class OddB(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 2, [1], {(0,0): "a", (0,1): "b", (1,1): "a", (1,0): "b"})

class EvenB(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 2, [0], {(0,0): "a", (0,1): "b", (1,1): "a", (1,0): "b"})

class FirstB(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 5, [1,2], {(0,1): "b", (1,1): "a", (1,2): "b", (2,2): "a", (2,1): "b", (0,3): "a", (3,3): "b", (3,4): "a", (4,4): "b", (4,3): "a"})

class FirstA(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 5, [1,2], {(0,1): "a", (1,1): "b", (1,2): "a", (2,2): "b", (2,1): "a", (0,3): "b", (3,3): "a", (3,4): "b", (4,4): "a", (4,3): "b"})

class LastB(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 2, [0], {(0,1): "a", (1,1): "a", (1,0): "b", (0,0): "b"})

class LastA(FiniteAutomaton):
    def __init__(self):
        super().__init__(["a", "b"], 2, [1], {(0,1): "a", (1,1): "b", (1,0): "a", (0,0): "b"})