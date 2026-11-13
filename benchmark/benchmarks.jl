# Env: benchmark. Run: julia -t 4 --project=benchmark benchmark/benchmarks.jl [quick]
using AdaptivePort, BenchmarkTools, Distributed, Printf

const QUICK = "quick" in ARGS
const RESULTS = Tuple{String, Float64}[]

function report(name, seconds, per = nothing)
    push!(RESULTS, (name, seconds))
    extra = per === nothing ? "" : @sprintf("  (%.2f us/point)", 1e6 * seconds / per)
    @printf("%-52s %10.4f s%s\n", name, seconds, extra)
    flush(stdout)
end

wiggle(x) = tanh(50x) + sin(3x)
wiggle(p::NTuple{2}) = tanh(50p[1] * p[2]) + sin(3p[1])

# Hand-written loop: ask one point, evaluate, tell.
function manual_loop!(l, n)
    while npoints(l) < n
        for x in ask(l, 1)
            tell!(l, x, l.f(x))
        end
    end
    return l
end

# Fixed ask/tell sequence used to check that optimizations keep the points identical.
function trace(l, n)
    pts = Any[]
    while npoints(l) < n
        for x in ask(l, 3)
            push!(pts, x)
            tell!(l, x, l.f(x))
        end
    end
    return pts, loss(l)
end

function bench_learners()
    # warm up compilation
    manual_loop!(Learner1D(wiggle; bounds = (-1, 1)), 50)
    manual_loop!(Learner2D(wiggle; bounds = ((-1, 1), (-1, 1))), 50)
    for n in (100, 1000, 10000)
        t = @elapsed manual_loop!(Learner1D(wiggle; bounds = (-1, 1)), n)
        report("Learner1D ask/tell loop, N = $n", t, n)
    end
    for n in (100, 1000, QUICK ? 1000 : 3000)
        t = @elapsed manual_loop!(Learner2D(wiggle; bounds = ((-1, 1), (-1, 1))), n)
        report("Learner2D ask/tell loop, N = $n", t, n)
    end
    l = Learner1D(wiggle; bounds = (-1, 1)); manual_loop!(l, 10000)
    report("Learner1D loss() at N = 10000 (x1000)", @elapsed(for _ in 1:1000 loss(l) end))
    l = Learner2D(wiggle; bounds = ((-1, 1), (-1, 1))); manual_loop!(l, 1000)
    report("Learner2D loss() at N = 1000 (x10)", @elapsed(for _ in 1:10 loss(l) end))
end

function bench_runner()
    n = 5000
    BlockingRunner(Learner1D(sin; bounds = (-1, 1)); max_npoints = 100)  # warm up compilation
    t0 = @elapsed manual_loop!(Learner1D(sin; bounds = (-1, 1)), n)
    t1 = @elapsed BlockingRunner(Learner1D(sin; bounds = (-1, 1)); max_npoints = n)
    report("1D hand loop, trivial f, N = $n", t0, n)
    report("1D BlockingRunner, trivial f, N = $n", t1, n)
    @printf("  runner overhead over hand loop: %.1f us/point\n", 1e6 * (t1 - t0) / n)
end

busy(x) = (t = time(); while time() - t < 0.01 end; sin(x))

function bench_parallel()
    n = 100
    BlockingRunner(Learner1D(busy; bounds = (-1, 1)); max_npoints = 10, ntasks = 2)  # warm up compilation
    t_seq = @elapsed BlockingRunner(Learner1D(busy; bounds = (-1, 1)); max_npoints = n)
    report("10 ms f, sequential, N = $n", t_seq)
    nt = Threads.nthreads()
    t_thr = @elapsed BlockingRunner(Learner1D(busy; bounds = (-1, 1)); max_npoints = n, ntasks = nt)
    report("10 ms f, threaded (ntasks = $nt), N = $n", t_thr)
    @printf("  speed-up %.2fx\n", t_seq / t_thr)
    pids = addprocs(2)
    try
        remotecall_eval(Main, pids, :(using AdaptivePort))
        remotecall_eval(Main, pids, :(busy(x) = (t = time(); while time() - t < 0.01 end; sin(x))))
        ex = DistributedExecutor(CachingPool(pids))
        BlockingRunner(Learner1D(busy; bounds = (-1, 1)); max_npoints = 10, executor = ex)  # warm up
        t_dist = @elapsed BlockingRunner(Learner1D(busy; bounds = (-1, 1)); max_npoints = n, executor = ex)
        report("10 ms f, Distributed (2 workers), N = $n", t_dist)
        @printf("  speed-up %.2fx\n", t_seq / t_dist)
    finally
        rmprocs(pids)
    end
end

function bench_balancing()
    if !isdefined(AdaptivePort, :BalancingLearner)
        println("BalancingLearner: not present in this checkout, skipped")
        return
    end
    for m in (10, 100)
        mk() = AdaptivePort.BalancingLearner([Learner1D(x -> sin(k * x); bounds = (-1, 1)) for k in 1:m])
        report("BalancingLearner, $m children, N = 2000", @elapsed(BlockingRunner(mk(); max_npoints = 2000)), 2000)
    end
end

function bench_save()
    n = QUICK ? 10^5 : 10^6
    l = Learner1D(sin; bounds = (-1, 1))
    for i in 1:n tell!(l, -1 + 2 * (i - 1) / (n - 1), sin(i)) end
    dir = mktempdir()
    for ext in ("csv", "jls")
        p = joinpath(dir, "d.$ext")
        save_data(p, l)
        report("save_data .$ext, N = $n", @elapsed(save_data(p, l)))
        report("load_data! .$ext, N = $n", @elapsed(load_data!(Learner1D(sin; bounds = (-1, 1)), p)))
    end
end

function main()
    println("threads = $(Threads.nthreads()), Julia $VERSION")
    sections = filter(a -> a in ("learners", "runner", "parallel", "balancing", "save"), ARGS)
    isempty(sections) && (sections = ["learners", "runner", "parallel", "balancing", "save"])
    "learners" in sections && bench_learners()
    "runner" in sections && bench_runner()
    "parallel" in sections && bench_parallel()
    "balancing" in sections && bench_balancing()
    "save" in sections && bench_save()
    if "trace" in ARGS
        for (name, l) in (("1d", Learner1D(wiggle; bounds = (-1, 1))), ("2d", Learner2D(wiggle; bounds = ((-1, 1), (-1, 1)))))
            pts, ls = trace(l, 300)
            println("TRACE $name ", hash(pts), " ", ls)
        end
    end
end
main()
