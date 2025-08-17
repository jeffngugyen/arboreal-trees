#!/usr/bin/env sage -python
# -*- coding: utf-8 -*-
"""
tree_resultant_cli.py

Purpose
-------
Given a polynomial p(x) ∈ ℚ[x] with a real root λ, set y := λ². This script:
  1) Computes the minimal polynomial M(y) of y via a resultant:
         M(y) := Res_x( p(x), x² - y )  (then made monic)
  2) Defines a small DSL of TREES (Leaf, UnaryLift, Merge) that produce a
     rational function F_T(y) by a recursive rule:
        - Leaf:         F(y) = 0
        - Merge kids:   F(y) = (1/y) * Σ_k [ 1 / (1 - F_child_k(y)) ]
        - UnaryLift U:  F(y) = 1 / ( y * (1 - F_U(y)) )
  3) Evaluates f_T(λ) := F_T(λ²) numerically and symbolically
  4) Tests the "hit" condition F_T(λ²) = 1 in TWO ways:
        (Exact)    G_T(y) := numerator(F_T(y) - 1)
                   Res_y( M(y), G_T(y) ) == 0  (certificate of shared root)
        (Numeric)  |F_T(λ²) - 1| ≤ tol
  5) Optionally SEARCHES over small trees (bounded depth/arity/pole size) to
     find a TREE that hits (constructive proof).

Command-line examples
---------------------
# Demo: golden ratio polynomial
sage -python tree_resultant_cli.py \
  --poly "x^2 - x - 1" --root max --symbolic --show-resultant --tree demo

# Constructive search for a hitting tree (exact resultant test)
sage -python tree_resultant_cli.py \
  --poly "x^2 - 2" --root max \
  --construct --depth 3 --mmax 6 --arity 3 --limit 8000 \
  --symbolic --show-resultant

Key mathematical facts
----------------------
• Minimal polynomial of y=λ² by resultant:
    If p has roots {α_i}, Res_x(p(x), x² - y) ∝ ∏_i (α_i² - y), so it vanishes
    exactly at y = α_i². For p ∈ ℚ[x], M(y) ∈ ℚ[y] and we make it monic.

• Tree recursion:
    F_T(y) ∈ ℚ(y). Numerically, we evaluate at y=λ² to get f_T(λ).

• Exact hit certificate:
    If F_T(λ²) = 1 and p(λ)=0, then y=λ² is a common root of M(y) and
    G_T(y)=num(F_T(y)-1). Computing Res_y(M,G_T) = 0 certifies the hit.
"""

from sage.all import *
import argparse
from functools import lru_cache
from itertools import combinations_with_replacement

# =============================================================================
#  Algebra: minimal polynomial of y = λ² via a resultant
# =============================================================================

def minimal_poly_of_lambda_sq(p):
    """
    Compute M(y) = Res_x( p(x), x^2 - y ) ∈ ℚ[y], made monic.

    Parameters
    ----------
    p : sage polynomial over QQ
        Univariate polynomial p(x) with rational coefficients.

    Returns
    -------
    M : sage polynomial over QQ in y
        Monic minimal polynomial of y = λ², where p(λ)=0.

    Rationale
    ---------
    The resultant eliminates x and produces a polynomial in y that vanishes
    precisely when y = α² for some root α of p.
    """
    R_x = p.parent()
    if R_x.base_ring() is not QQ:
        raise TypeError("p must be over QQ")
    R_y = PolynomialRing(QQ, 'y');  y = R_y.gen()
    S   = PolynomialRing(R_y, 'x'); xS = S.gen()
    res = S(p).resultant(xS**2 - y)
    return R_y(res.monic())

# =============================================================================
#  Tree DSL
# =============================================================================

class Leaf:
    """
    Leaf node.
    Semantics: F_leaf(y) = 0.
    """
    def __repr__(self): return "Leaf()"

class UnaryLift:
    """
    UnaryLift(child).
    Semantics: F(y) = 1 / ( y * (1 - F_child(y)) ).
    This is the “self-energy” update of attaching a parent above one child.
    """
    def __init__(self, child): self.child = child
    def __repr__(self): return f"UnaryLift({self.child!r})"

class Merge:
    """
    Merge([T1, T2, ..., Td]).
    Semantics: F(y) = (1/y) * Σ_{k=1}^d 1 / (1 - F_Tk(y)).
    Attaches a parent that merges multiple child branches.
    """
    def __init__(self, children): self.children = list(children)
    def __repr__(self): return f"Merge({self.children!r})"

# -----------------------------------------------------------------------------
# Numeric and symbolic evaluators for F_T
# -----------------------------------------------------------------------------

def f_value(lam, node):
    """
    Numeric evaluator: f_T(λ) = F_T(λ²).

    Parameters
    ----------
    lam : real algebraic number (AA) or numeric
        The chosen real root λ of p(x).
    node : Leaf | UnaryLift | Merge
        The tree to evaluate.

    Returns
    -------
    sage number (exact or AA)
        The numeric value f_T(λ).
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

def F_symbolic(node):
    """
    Symbolic evaluator: F_T(y) ∈ ℚ(y).

    Returns
    -------
    Fraction over QQ[y]
        The rational function F_T(y), computed recursively.
    """
    Ry = FractionField(PolynomialRing(QQ, 'y')); y = Ry.gen()
    if isinstance(node, Leaf):
        return Ry(0)
    if isinstance(node, UnaryLift):
        a = F_symbolic(node.child)
        return 1 / ( y * (1 - a) )
    if isinstance(node, Merge):
        s = Ry(0)
        for ch in node.children:
            a = F_symbolic(ch)
            s += 1 / (1 - a)
        return s / y
    raise TypeError("Unknown node type")

def as_common_numden(rat):
    """
    Normalize a rational function to coprime numerator/denominator over QQ[y].

    Parameters
    ----------
    rat : Fraction in QQ(y)

    Returns
    -------
    (num, den) : (QQ[y], QQ[y])
        Numerator and denominator with gcd(num,den)=1.

    Usage
    -----
    G_T(y) := numerator(F_T(y) - 1)  (the 'hit equation' polynomial).
    """
    par  = rat.parent()      # FractionField(QQ[y])
    Rpy  = par.base_ring()   # QQ[y]
    num  = rat.numerator()   # QQ[y]
    den  = rat.denominator() # QQ[y]
    g    = gcd(num, den)
    return (num // g, den // g)

# -----------------------------------------------------------------------------
# Handy building blocks
# -----------------------------------------------------------------------------

def simple_pole(m):
    """
    A unary-lifted star with m leaves.
    Derivation:
      Merge([Leaf]*m) has F = m / y  (since each Leaf contributes 1/(1-0)=1).
      UnaryLift( Merge(...) ) gives F = 1 / ( y * (1 - m/y) ) = 1 / (y - m).
    """
    return UnaryLift(Merge([Leaf() for _ in range(m)]))

def demo_tree():
    """
    A tiny example:
      Merge([UnaryLift(Leaf()), UnaryLift(Leaf())])
    Symbolically:
      UnaryLift(Leaf) has F = 1 / (y*(1-0)) = 1/y
      Merge of two gives F = (1/y)*(1/(1-1/y)+1/(1-1/y)) = 2/(y-1).
    """
    return Merge([UnaryLift(Leaf()), UnaryLift(Leaf())])

# =============================================================================
#  Pretty-printing of trees (ASCII)
# =============================================================================

def pretty_tree(node, indent="", last=True):
    """
    ASCII pretty-printer for Leaf / UnaryLift / Merge trees.

    Example output (for unary-lifted 3-star):
        └─ UnaryLift
           └─ Merge[3]
              ├─ Leaf
              ├─ Leaf
              └─ Leaf
    """
    branch = "└─ " if last else "├─ "
    if isinstance(node, Leaf):
        return indent + branch + "Leaf\n"
    if isinstance(node, UnaryLift):
        s = indent + branch + "UnaryLift\n"
        s += pretty_tree(node.child, indent + ("   " if last else "│  "), True)
        return s
    if isinstance(node, Merge):
        s = indent + branch + f"Merge[{len(node.children)}]\n"
        for i, ch in enumerate(node.children):
            s += pretty_tree(ch, indent + ("   " if last else "│  "),
                             i == len(node.children)-1)
        return s
    raise TypeError("Unknown node type in pretty_tree")

# =============================================================================
#  Parsing polynomials and choosing a root
# =============================================================================

def parse_poly(poly_str):
    """
    Parse a string like "x^2 - x - 1" into a monic polynomial p(x) ∈ ℚ[x].
    """
    R = PolynomialRing(QQ, 'x'); x = R.gen()
    try:
        p = R(SR(poly_str).subs(x=x))
    except Exception as e:
        raise ValueError(f"Could not parse polynomial: {poly_str!r}") from e
    if p.degree() <= 0:
        raise ValueError("Polynomial must have degree ≥ 1.")
    lc = p.leading_coefficient()
    return p if lc == 1 else (p / lc)

def choose_root(p, mode="max", index=None, prefer="AA"):
    """
    Choose a real root λ of p(x).

    Parameters
    ----------
    p : polynomial over QQ
    mode : {"max","min","index"}
        Which real root to pick (largest, smallest, or by 0-based index).
    index : int or None
        Used if mode="index".
    prefer : {"AA","QQbar"}
        - "AA": only real algebraic roots (fast, clean).
        - "QQbar": all algebraic roots; filter by imag()==0.

    Returns
    -------
    λ : algebraic real (AA) or QQbar element (real)
    """
    if prefer.upper() == "AA":
        real_roots = [r for r, _ in p.roots(ring=AA)]
    else:
        roots = [r for r, _ in p.roots(ring=QQbar)]
        real_roots = [r for r in roots if r.imag().is_zero()]
    if not real_roots:
        raise ValueError("No real roots found.")
    if mode == "max":
        try:   return max(real_roots)
        except TypeError: return max(real_roots, key=lambda z: RR(z))
    if mode == "min":
        try:   return min(real_roots)
        except TypeError: return min(real_roots, key=lambda z: RR(z))
    if mode == "index":
        if index is None or not (0 <= index < len(real_roots)):
            raise ValueError(f"--root-index must be in [0, {len(real_roots)-1}]")
        ordered = sorted(real_roots, key=lambda z: RR(z))
        return ordered[index]
    raise ValueError(f"Unknown root mode: {mode}")

# =============================================================================
#  Tree-spec (mini) parser for CLI convenience
# =============================================================================

def parse_tree_spec(spec):
    """
    Parse a recursive spec into a tree.

    Atoms:
      leaf               → Leaf()
      demo               → demo_tree()
      pole:m             → simple_pole(m) = UnaryLift(Merge(Leaf^m))
      merge_poles:m1,m2  → Merge([pole(m1), pole(m2), ...])

    Recursive forms:
      lift:<spec>               → UnaryLift(parse(spec))
      merge:<s1>|<s2>|...       → Merge([parse(s1), parse(s2), ...])

    Examples
    --------
    "demo"                       → Merge([UnaryLift(Leaf()), UnaryLift(Leaf())])
    "pole:3"                     → UnaryLift(Merge([Leaf(),Leaf(),Leaf()]))
    "merge_poles:1,2,4"          → Merge([pole(1), pole(2), pole(4)])
    "lift:merge:pole:1|lift:pole:2"
                                → UnaryLift(Merge([pole(1), UnaryLift(pole(2))]))
    """
    spec = spec.strip()
    if spec == "leaf": return Leaf()
    if spec == "demo": return demo_tree()
    if spec.startswith("pole:"):
        m = int(spec.split(":", 1)[1]);  return simple_pole(m)
    if spec.startswith("merge_poles:"):
        ms = [int(t) for t in spec.split(":",1)[1].split(",") if t.strip()]
        return Merge([ simple_pole(m) for m in ms ])
    if spec.startswith("lift:"):
        inner = spec.split(":", 1)[1];   return UnaryLift(parse_tree_spec(inner))
    if spec.startswith("merge:"):
        rest = spec.split(":", 1)[1]
        parts = [s for s in rest.split("|") if s.strip()]
        return Merge([ parse_tree_spec(s) for s in parts ])
    raise ValueError(f"Unrecognized tree spec: {spec!r}")

# =============================================================================
#  Enumeration + caching for constructive search
# =============================================================================

def tree_size(node):
    """Count nodes (Leaf/UnaryLift/Merge) for a simple size metric."""
    if isinstance(node, Leaf): return 1
    if isinstance(node, UnaryLift): return 1 + tree_size(node.child)
    if isinstance(node, Merge): return 1 + sum(tree_size(ch) for ch in node.children)
    raise TypeError("Unknown node type")

def serialize_tree(node):
    """
    Canonical serialization (string) used for caching F_T(y).

    Examples
    --------
    Leaf()                             → "leaf"
    UnaryLift(Leaf())                  → "lift(leaf)"
    Merge([Leaf(), Leaf()])            → "merge(leaf|leaf)"  (sorted children)
    """
    if isinstance(node, Leaf): return "leaf"
    if isinstance(node, UnaryLift): return f"lift({serialize_tree(node.child)})"
    if isinstance(node, Merge):
        parts = sorted(serialize_tree(ch) for ch in node.children)
        return "merge(" + "|".join(parts) + ")"
    raise TypeError("Unknown node type")

@lru_cache(maxsize=None)
def F_symbolic_from_serial(serial):
    """
    Cached symbolic evaluation: serial → tree → F_T(y).
    Avoids recomputing F_T for identical subtrees encountered via enumeration.
    """
    # a tiny deserializer:
    def tokens(s):
        i, n = 0, len(s)
        while i < n:
            c = s[i]
            if c.isspace():
                i += 1
            elif c in '(),|':
                yield c; i += 1
            else:
                j = i
                while j < n and (s[j].isalnum() or s[j] in '_:'):
                    j += 1
                yield s[i:j]; i = j

    def parse_stream(ts):
        tok = next(ts)
        if tok == "leaf": return Leaf()
        if tok == "lift":
            assert next(ts) == "("
            child = parse_stream(ts)
            assert next(ts) == ")"
            return UnaryLift(child)
        if tok == "merge":
            assert next(ts) == "("
            chs = []
            while True:
                chs.append(parse_stream(ts))
                sep = next(ts)
                if sep == ")": break
                assert sep == "|"
            return Merge(chs)
        raise ValueError(f"Bad token: {tok}")

    T = parse_stream(iter(tokens(serial)))
    return F_symbolic(T)

def all_trees(depth, mmax=6, max_arity=3):
    """
    Enumerate small trees up to given recursion depth.

    Strategy
    --------
    depth=0:
        base atoms = {Leaf} ∪ {pole:m | 1 ≤ m ≤ mmax}
    depth>0:
        from all previous pools, generate UnaryLift(...) and Merge of
        arity 2..max_arity (combinations_with_replacement to keep order canonical).

    Yields
    ------
    Trees in increasing "complexity" order, which helps find small hits first.
    """
    base = [Leaf()] + [simple_pole(m) for m in range(1, mmax+1)]
    # first, the base
    for t in base:
        yield t
    if depth <= 0: return
    pools = [base]
    for _ in range(1, depth+1):
        new_level = []
        # Unary lifts of the last level
        for T in pools[-1]:
            new_level.append(UnaryLift(T))
        # Merges pulling children from all earlier pools (richer combos)
        soup = []
        for P in pools:
            soup.extend(P)
        for ar in range(2, max_arity+1):
            for kids in combinations_with_replacement(soup, ar):
                new_level.append(Merge(kids))
        pools.append(new_level)
        for T in new_level:
            yield T

# =============================================================================
#  Hit tests (exact resultant and numeric fallback)
# =============================================================================

def hit_via_resultant(M, F, lam_sq):
    """
    Exact certificate: Let G(y)=numerator(F(y)-1).
    Hit iff Res_y(M(y), G(y)) == 0.

    Returns
    -------
    (is_hit, G, resultant_value)
      is_hit : bool
      G     : QQ[y] polynomial numerator(F-1)
      resultant_value : integer/rational (0 indicates a shared root)
    """
    num, den = as_common_numden(F - 1)
    Rval = M.resultant(num)
    return (Rval == 0, num, Rval)

def hit_via_numeric(F, lam_sq, tol=1e-12):
    """
    Numeric test: check whether |F(λ²) - 1| ≤ tol.
    Useful as a fast filter or when exact algebra is too costly.

    Returns
    -------
    (is_hit, residual)
      is_hit   : bool
      residual : float (|F(λ²)-1|)
    """
    try:
        res = abs(N(F(lam_sq) - 1))
        return (res <= tol, res)
    except Exception:
        return (False, float('inf'))

def search_tree_hits(p, lam, M, depth=3, mmax=6, max_arity=3, limit=5000,
                     numeric_only=False, tol=1e-12, verbose_every=500,
                     stop_at_first=True):
    """
    Constructive search for trees T hitting F_T(λ²)=1.

    Parameters
    ----------
    p, lam, M : as above
    depth, mmax, max_arity, limit : enumeration budget knobs
    numeric_only : if True, skip resultant in search loop (use numeric only)
    tol : numeric tolerance
    verbose_every : status print frequency
    stop_at_first : if True, return as soon as the first hit is found

    Returns
    -------
    list of hits (possibly length 0 or 1 if stop_at_first=True).
    Each hit is a dict with:
      - "tree"       : the tree object
      - "serial"     : its canonical string form
      - "F"          : symbolic F_T(y)
      - "G"          : numerator(F_T(y)-1) ∈ QQ[y]
      - "resultant"  : Res_y(M,G) (0 indicates exact shared root)
      - "residual"   : |F_T(λ²)-1| (numeric check)
      - "size"       : number of nodes (Leaf/UnaryLift/Merge)
    """
    yval = lam**2
    checked = 0
    hits = []
    for T in all_trees(depth=depth, mmax=mmax, max_arity=max_arity):
        checked += 1
        if verbose_every and checked % verbose_every == 0:
            print(f"[search] examined {checked} trees...")
        s = serialize_tree(T)
        F = F_symbolic_from_serial(s)

        ok, resid, G, Rval = False, float('inf'), None, None

        # Exact resultant unless numeric-only
        if not numeric_only:
            is_hit, G, Rval = hit_via_resultant(M, F, yval)
            ok = is_hit
            resid = 0.0 if ok else resid

        # Numeric fallback (or only test if numeric-only)
        if not ok:
            ok, resid = hit_via_numeric(F, yval, tol=tol)
            if ok:
                # Compute G/Rval post-hoc (optional; nice for reporting)
                G, _ = as_common_numden(F - 1)
                Rval = M.resultant(G)

        if ok:
            hits.append({
                "tree": T, "serial": s, "F": F, "G": G,
                "resultant": Rval, "residual": float(resid),
                "size": tree_size(T),
            })
            if stop_at_first:
                return hits
        if checked >= limit:
            break
    return hits

# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Resultant + Tree recursion: compute M(y) for y=λ² and evaluate/search F_T(λ²).")
    ap.add_argument("--poly", required=True, help='Polynomial in x over QQ, e.g. "x^2 - x - 1"')
    ap.add_argument("--root", default="max", choices=["max","min","index"],
                    help="Which real root of p(x) to use as λ.")
    ap.add_argument("--root-index", type=int, default=None,
                    help="If --root index, which index among real roots (0-based).")
    ap.add_argument("--prefer-root-ring", choices=["AA", "QQbar"], default="AA",
                    help="Ring for root selection (AA=real only, QQbar=complex then filter).")
    ap.add_argument("--show-roots", action="store_true",
                    help="Print the real roots found.")
    ap.add_argument("--symbolic", action="store_true",
                    help="Also print F_T(y) and G_T(y)=numerator(F_T(y)-1).")
    ap.add_argument("--show-resultant", action="store_true",
                    help="Compute and print Res_y( M(y), G_T(y) ).")
    ap.add_argument("--tree", default="demo",
                    help=("Tree spec. Options: demo | leaf | pole:m | merge_poles:m1,m2 | "
                          "lift:<spec> | merge:<spec1>|<spec2>|..."))

    # Constructive search options
    ap.add_argument("--construct", action="store_true",
                    help="Search for a small tree T with F_T(λ^2)=1.")
    ap.add_argument("--exhaustive", action="store_true",
                    help="Collect and print all hits found within the search budget.")
    ap.add_argument("--depth", type=int, default=3,
                    help="Max recursive depth for search (default: 3).")
    ap.add_argument("--mmax", type=int, default=6,
                    help="Max m for pole:m gadgets (default: 6).")
    ap.add_argument("--arity", type=int, default=3,
                    help="Max children in Merge(...) during search (default: 3).")
    ap.add_argument("--limit", type=int, default=5000,
                    help="Max number of trees to examine (default: 5000).")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="Numeric tolerance for |F_T(λ^2)-1| (fallback).")
    ap.add_argument("--numeric-only", action="store_true",
                    help="Skip exact resultant in search; numeric test only.")
    args = ap.parse_args()

    # Parse polynomial, choose λ, compute M(y)
    p   = parse_poly(args.poly)
    lam = choose_root(p, mode=args.root, index=args.root_index, prefer=args.prefer_root_ring)
    M   = minimal_poly_of_lambda_sq(p)

    print(f"p(x) = {p}")
    if args.show_roots:
        aa_roots = [r for r,_ in p.roots(ring=AA)]
        print("real roots (AA):", [N(r, 50) for r in aa_roots])
    print("λ ≈", N(lam, 50))
    print("Check p(λ) ≈", N(p(lam), 50))
    print("M(y) for y=λ^2:", M)
    print("Check M(λ^2) ≈", N(M(lam**2), 50))

    # -----------------------------
    # Constructive search for a hit
    # -----------------------------
    if args.construct:
        print("\n[construct] searching for a small tree with F_T(λ^2)=1 ...")
        hits = search_tree_hits(
            p, lam, M,
            depth=args.depth, mmax=args.mmax, max_arity=args.arity, limit=args.limit,
            numeric_only=args.numeric_only, tol=args.tol, verbose_every=500,
            stop_at_first=not args.exhaustive
        )
        if not hits:
            print("[construct] no hit within the search budget.")
        else:
            # Sort hits by a simple "simplicity" metric
            hits.sort(key=lambda h: (h["size"], len(h["serial"]), h["residual"]))
            for idx, hit in enumerate(hits, 1):
                print(f"\n[construct] HIT #{idx}")
                print("Tree:", hit["tree"])
                print("Tree (ASCII):\n" + pretty_tree(hit["tree"]))
                print("Serial:", hit["serial"])
                print("F_T(y) =", hit["F"])
                print("G_T(y) = numerator(F-1) =", hit["G"])
                print("Resultant Res_y(M,G) =", hit["resultant"])
                print("Residual |F_T(λ^2)-1| ≈", hit["residual"])
                print("Numeric check F_T(λ^2) ≈", N(hit["F"](lam**2), 50))
        return

    # -----------------------------
    # Single-tree evaluation
    # -----------------------------
    T   = parse_tree_spec(args.tree)
    val = f_value(lam, T)
    print("\nTree:", T)
    print("Tree (ASCII):\n" + pretty_tree(T))
    print("f_T(λ) = F_T(λ^2) ≈", N(val, 50))

    if args.symbolic or args.show_resultant:
        F = F_symbolic(T)
        num, den = as_common_numden(F - 1)
        print("\nSymbolic F_T(y) =", F)
        print("G_T(y) := numerator(F_T(y) - 1) =", num)
        print("Check F_T(λ^2) - 1 ≈", N(F(lam**2) - 1, 50))
        if args.show_resultant:
            R = M.resultant(num)
            print("\nRes_y( M(y), G_T(y) ) =", R)
            print("Res_y(M,G_T) == 0 ?", R == 0)
            if R == 0:
                print("\n==> Exact hit: F_T(λ^2) = 1 (shared root with M).")

if __name__ == "__main__":
    main()

