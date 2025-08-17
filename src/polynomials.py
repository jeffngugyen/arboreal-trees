# src/polynomials.py
from sage.all import *

def minimal_poly_of_lambda_sq(p):
    """
    Given p(x) ∈ QQ[x], return M(y) ∈ QQ[y] where y = λ^2 and λ is a root of p.
    Computes M(y) = Res_x( p(x), x^2 - y ).
    """
    # Ensure p is univariate over QQ
    R_x = p.parent()
    if R_x.base_ring() is not QQ:
        raise TypeError("p must be over QQ")
    x = R_x.gen()

    # Build QQ[y] then S = (QQ[y])[x]
    R_y = PolynomialRing(QQ, 'y')
    y = R_y.gen()
    S = PolynomialRing(R_y, 'x')
    xS = S.gen()

    # View p in S and compute resultant in x
    p_in_S = S(p)                      # embed p(x) into S
    res = p_in_S.resultant(xS**2 - y) 

    # Result is in R_y (QQ[y]); return monic for cleanliness
    return res.monic()

