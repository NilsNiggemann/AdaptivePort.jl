# Constants of the doubly adaptive Clenshaw-Curtis scheme (Gonnet's algorithm 4), computed once at load.
# Port of upstream's `integrator_coeffs.py`. Exact rational arithmetic (BigInt) is used where upstream uses
# `fractions.Fraction`. Arrays are 1-based: the entry of refinement depth `d` (0 to 3) is at index `d + 1`.
module Coeffs

import LinearAlgebra

# Legendre polynomials (standard normalisation) up to degree `n - 1`, as coefficient vectors in the monomials.
function legendre(n)
    result = [Rational{BigInt}[1], Rational{BigInt}[0, 1]]
    n <= 2 && return result[1:n]
    for i in 2:n-1
        new = zeros(Rational{BigInt}, i + 1)
        new[2:end] .= result[end] .* (2i - 1)
        new[1:i-1] .-= result[end-1] .* (i - 1)
        push!(result, new ./ i)
    end
    return result
end

# Monomial coefficients of the Newton polynomial over the `n` nodes -cos(i pi / (n - 1)). The coefficients are
# sums of products of cosines, which are collected exactly in `terms` using cos(a)cos(b) = (cos(a+b) + cos(a-b))/2.
function newton(n)
    modulus = 2 * (n - 1)
    terms = Dict{Tuple{Int,Int},Int}((0, 0) => 1)
    for i in 0:n-1
        newterms = Tuple{Int,Int,Int}[]
        for ((d, a), m) in terms, b in (i, -i)
            arg = mod(a + b, modulus)
            arg > n - 1 && (arg = modulus - arg)
            if arg >= n ÷ 2
                isodd(n) && arg == n ÷ 2 && continue
                push!(newterms, (d + 1, n - 1 - arg, -m))
            else
                push!(newterms, (d + 1, arg, m))
            end
        end
        for (d, s, m) in newterms
            terms[(d, s)] = get(terms, (d, s), 0) + m
        end
    end
    c = zeros(Int, n + 1)
    for ((d, a), m) in terms
        m != 0 && a != 0 && error("Newton polynomial cannot be represented exactly.")
        c[n-d+1] += m
    end
    return c ./ 2.0 .^ (n:-1:0)
end

# Integral over [-1, 1] of the product of two polynomials given by monomial coefficients.
function scalar_product(a, b)
    la = length(a)
    c = zeros(eltype(a), length(b) + la + 1)
    for (i, bi) in enumerate(b), j in (i-1)%2:2:la-1
        iszero(bi) && continue
        c[i+j] += a[j+1] * bi
    end
    return 2 * sum(c[i+1] / (i + 1) for i in 0:2:length(c)-1)
end

# Decomposition of the Newton polynomials in orthonormal Legendre polynomials, for each rule size in `ns`.
function calc_bdef(ns)
    legs = legendre(maximum(ns) + 1)
    return map(ns) do n
        a = Rational{BigInt}.(newton(n))
        [sqrt((2length(b) - 1) / 2) * Float64(scalar_product(a, b)) for b in legs[1:n+1]]
    end
end

# Vandermonde-like matrix of the orthonormal Legendre polynomials at the nodes `x` (rows: nodes).
function calc_V(x, n)
    V = [ones(length(x)), copy(x)]
    for i in 2:n-1
        push!(V, (2i - 1) / i .* x .* V[end] .- (i - 1) / i .* V[end-1])
    end
    for i in 0:n-1
        V[i+1] .*= sqrt(i + 0.5)
    end
    return reduce(hcat, V)
end

const NS = (5, 9, 17, 33)
const EPS = eps(Float64)

# Nodes, made exactly antisymmetric (important for splitting intervals). Same arithmetic as `numpy.linspace`.
const XI = map(NS) do n
    x = [-cos(k * (pi / (n - 1))) for k in 0:n-1]
    x[end] = -cos(pi)
    (x .- reverse(x)) ./ 2
end

const V = [calc_V(x, n) for (x, n) in zip(XI, NS)]
const V_INV = [LinearAlgebra.inv(v) for v in V]
const VCOND = [LinearAlgebra.opnorm(a, 2) * LinearAlgebra.opnorm(b, 2) for (a, b) in zip(V, V_INV)]

# Shift matrices: coefficients of the left and right half of an interval from those of the whole.
const T_LEFT, T_RIGHT = (V_INV[4] * calc_V((XI[4] .+ a) ./ 2, NS[4]) for a in (-1, 1))

# Relative difference of two consecutive approximations below which the error estimate is deemed reliable.
const HINT = 0.1
# Smallest acceptable relative separation of points in a rule.
const MIN_SEP = 16 * EPS
# Number of divergent-looking splits tolerated.
const NDIV_MAX = 20

# Downdate matrix (algorithm 5 of Gonnet's thesis); `GAMMA[1:2]` are unused.
const ALPHA = [sqrt((k + 1)^2 / (2k + 1) / (2k + 3)) for k in 0:NS[4]-1]
const GAMMA = [k < 2 ? 0.0 : sqrt(k^2 / (4k^2 - 1)) for k in 0:NS[4]-1]
const B_DEF = calc_bdef(NS)

end # module Coeffs
