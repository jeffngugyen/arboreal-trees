from sage.all import *

class Leaf: 
    pass

class UnaryLift:
    def __init__(self, child):
        self.child = child

class Merge:
    def __init__(self, children):
        self.children = list(children)

def f_value(lam, node):
    """Compute f_T(lam) via:
       Leaf: 0
       UnaryLift: 1 / (lam^2 * (1 - f_child))
       Merge: (1/lam^2) * sum_k 1/(1 - f_child_k)
    """
    if isinstance(node, Leaf):
        return QQ(0)
    if isinstance(node, UnaryLift):
        a = f_value(lam, node.child)
        return 1 / (lam**2 * (1 - a))
    if isinstance(node, Merge):
        s = QQ(0)
        for ch in node.children:
            a = f_value(lam, ch)
            s += 1 / (1 - a)
        return s / (lam**2)
    raise TypeError("Unknown node type")

def simple_pole(lam, m):
    """Realize 1/(lam^2 - m) as UnaryLift(Merge([Leaf]*m))."""
    merged = Merge([Leaf() for _ in range(m)])  # f = m/lam^2
    return UnaryLift(merged)                    # f = 1/(lam^2 - m)
