using NanotubesCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic
using RadiiPolynomial

using JLD2

### Easier to converge with parameters equal to 1
b = 1
θ = 2π / 3
kb = 1
kθ = 1

connectivity, x_initial = get_5_5_connectivity_even(30)
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)

### BFGS
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, x_initial, method=algo, g_tol=1e-6; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

### Newton refinement
F = x -> extended_Grad(x, x_BFGS, connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_BFGS, connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-13, maxiter=10)[1]

### Convergence with carbon parameters
b = 1.44
θ = 2π / 3
kb = 469
kθ = 63

F = x -> extended_Grad(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
DF = x -> extended_Hess(x, x_newton[7:end], connectivity, b, θ, kb, kθ)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_newton[7:end], :, 1)]; tol=1.0e-13, maxiter=10)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]

### Proof 
F_int = x -> extended_Grad(x, interval.(x_newton[7:end]), connectivity, interval(b), interval(θ), interval(kb), interval(kθ))
DF_int = x -> extended_Hess(x, interval.(x_newton[7:end]), connectivity, interval(b), interval(θ), interval(kb), interval(kθ))

# find_maximum_r_star_test(x_newton, F_int, DF_int, 9.998779363575441e-8, 1e-7)
# r = find_maximum_r_star(x_newton, F_int, DF_int)

r = get_proof(x_newton, F_int, DF_int, 9.999931334517948e-7)

### Save the data 
save("data/5-5_PentCaps_N360_Harmonic.jld2", "x_newton", x_newton, "r", r)

