# scripts/build_tree_from_poly.py
from sage.all import *
import os, sys

# Make the project root importable (so "src" is found)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.polynomials import minimal_poly_of_lambda_sq
from src.tree_build import Leaf, UnaryLift, Merge, f_value, simple_pole


def demo_tree(lam):
    """Example tree: two unary-lifted leaves merged → f_T = 2 / lam^2."""
    return Merge([UnaryLift(Leaf()), UnaryLift(Leaf())])


def main():
    # Use script-safe Sage style (no R.<x> sugar)
    R = PolynomialRing(QQ, 'x')
    x = R.gen()

    # Example polynomial: x^2 - x - 1 (golden ratio root)
    p = x**2 - x - 1

    # Pick a real root (AA = real algebraic field)
    lam = p.roots(ring=AA)[0][0]

    # Minimal polynomial of y = lam^2
    M = minimal_poly_of_lambda_sq(p)

    print("p(x) =", p)
    print("lambda ≈", N(lam, 30))
    print("M(y) for y = lambda^2:", M)

    # Build demo tree and evaluate f_T(lambda)
    T = demo_tree(lam)
    val = f_value(lam, T)
    print("f_T(lambda) ≈", N(val, 30))


if __name__ == "__main__":
    main()

