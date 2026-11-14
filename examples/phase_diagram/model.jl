# Toy model for the phase-diagram example: mean-field transverse-field Ising magnet with two competing orders.
# The coupling ratio g moves weight from the coupling J_A = 1 - g, which favours order A, to J_B = g, which
# favours order B. A transverse field GAMMA disorders both, so near g = 1/2 neither orders, even at T = 0.
# Phases: A (m_A > 0), B (m_B > 0), disordered (m_A = m_B = 0). Transitions are continuous (m ~ sqrt).

const GAMMA = 0.6  # transverse field; order needs J > GAMMA at T = 0

"Mean-field order parameter of a transverse-field Ising magnet with coupling `J` at temperature `T`."
function order_parameter(J, T; tol = 1e-12)
    # self-consistency m = (J m / h) tanh(h / T) with h = sqrt((J m)^2 + GAMMA^2); F decreases in m
    F(m) = J / hypot(J * m, GAMMA) * tanh(hypot(J * m, GAMMA) / T) - 1
    F(0.0) <= 0 && return 0.0  # only the disordered solution
    lo, hi = 0.0, 1.0
    while hi - lo > tol
        mid = (lo + hi) / 2
        F(mid) > 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

"Order parameters `(m_A, m_B)` at coupling ratio `g` and temperature `T`; `cost` seconds of sleep mimic an expensive solver."
function order_parameters(g, T; cost = 0.0)
    cost > 0 && sleep(cost)
    return (order_parameter(1 - g, T), order_parameter(g, T))
end
