# Uniform vs adaptive sampling of three sharp functions, same number of evaluations.
# Each function is rescaled to [0, 1] (by its range on a fine grid) so all share the axes.
# Run with: julia --project=examples examples/learner1d_plot.jl
using AdaptivePort, CairoMakie, MakieHelpers

# Peak of width a on a linear background (upstream Adaptive's standard example).
const a = 0.01
peak(x) = x + a^2 / (a^2 + x^2)
peaks3(x) = sum(a^2 / (a^2 + (x - c)^2) for c in (-0.5, 0.0, 0.5))
expo(x) = sin(4pi * x)

bounds = (-1.0, 1.0)
budget = 50

function adaptive_samples(f; bounds, budget)
    learner = Learner1D(f; bounds)
    while npoints(learner) < budget
        for x in ask(learner, 1)
            tell!(learner, x, f(x))
        end
    end
    return result(learner)
end

x_uniform = range(bounds...; length = budget)
x_fine = range(bounds...; length = 4001)
# (function, color, label)
funcs = ((expo, :blue, L"0.5\sin(8x)" , 0.5), (peaks3, :red, L"\textrm{3 peaks}", 1.0),(peak, :black, L"\textrm{peak}", 1.0))
fig = with_theme(theme_SimpleTicks(; publication = true)) do
    fig = Figure(size = (680, 360))
    ax1 = Axis(fig[1, 1]; ylabel = L"(f - f_{\min})/(f_{\max} - f_{\min})", xticklabelsvisible = false)
    ax2 = Axis(fig[1, 2]; xticklabelsvisible = false, yticklabelsvisible = false)
    linkaxes!(ax1, ax2)
    # point-spacing density 1/Δx, drawn at interval midpoints
    ax3 = Axis(fig[2, 1]; xlabel = L"x", ylabel = L"1/\Delta x", yscale = log10)
    ax4 = Axis(fig[2, 2]; xlabel = L"x", yscale = log10, yticklabelsvisible = false)
    linkxaxes!(ax1, ax2, ax3, ax4)
    linkyaxes!(ax3, ax4)
    rowsize!(fig.layout, 2, Relative(0.3))
    for (f, color, lab, s) in funcs
        lo, hi = extrema(f.(x_fine))
        g(y) = (y - lo) / (hi - lo) * s
        data = adaptive_samples(f; bounds, budget)
        for (ax, axd, xs, ys) in ((ax1, ax3, x_uniform, f.(x_uniform)), (ax2, ax4, data.x, data.y))
            lines!(ax, x_fine, g.(f.(x_fine)); color = (color, 0.3))
            scatterlines!(ax, xs, g.(ys); color, markersize = 5)
            xs = sort(collect(xs))
            stairs!(axd, xs, [1 ./ diff(xs); NaN]; step = :post, color)
        end
    end
    panellabel!(ax1, L"(a)\; \textrm{uniform}")
    panellabel!(ax2, L"(b)\; \textrm{Learner1D}")
    entries = [[LineElement(color = c), MarkerElement(marker = :circle, color = c)] for (_, c, _) in funcs]
    axislegend(ax1, entries, [n for (_, _, n) in funcs]; position = :lt, framevisible = false, margin = (8, 8, 30, 8))
    colgap!(fig.layout, 8)
    # save("learner1d_vs_uniform.pdf", fig)
    fig
end
save(joinpath(@__DIR__, "learner1d_vs_uniform.png"), fig; px_per_unit = 1)
fig
