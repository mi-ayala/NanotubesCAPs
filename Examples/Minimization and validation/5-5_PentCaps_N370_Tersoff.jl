using NanotubesCAPs

using LineSearches
using ForwardDiff
using Optim

using IntervalArithmetic

using JLD2

### A proof of the (5,5)-nanotube with pentagonal caps and N = 370 atoms using the Tersoff potential. We minimize the energy and validate the simulation. All the other examples follow the same implementation structure.


### We start by minimizing the harmonic energy. Before using Tersoff potential.
b = 1.4
θ = 2π / 3
kb = 1
kθ = 1

p = Tersoff_parameters()

connectivity, x_initial = get_5_5_connectivity_odd(33)
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

### BFGS optimization stage and Newton refinement 
e = x -> Tersoff_energy(x, p, connectivity)
algo = LBFGS(; m=30, alphaguess=LineSearches.InitialStatic(), linesearch=LineSearches.BackTracking())
res = Optim.optimize(e, center_nanotube_armchair(x_newton[7:end]), method=algo; autodiff=:forward)
x_BFGS = vec(reshape(Optim.minimizer(res), :, 1))
x_BFGS = center_nanotube_armchair(x_BFGS)

F = x -> extended_Grad_Tersoff(x, x_BFGS, p, connectivity)
DF = x -> extended_Hess_Tersoff(x, x_BFGS, p, connectivity)
x_newton = newton_method(x -> (F(x), DF(x)), [zeros(6); reshape(x_BFGS, :, 1)]; tol=1.0e-12, maxiter=10)[1]

x_newton = [zeros(6); reshape(center_nanotube_armchair(x_newton[7:end]), :, 1)]


### Validation step using interval arithmetic.
F_int = x -> extended_Grad_Tersoff(x, interval.(x_newton[7:end]), p, connectivity)
DF_int = x -> extended_Hess_Tersoff(x, interval.(x_newton[7:end]), p, connectivity)

r = get_proof(x_newton, F_int, DF_int, 9.9212e-8)

# ### Save the data 
# save("data/5-5_PentCaps_N370_Tersoff.jld2", "x_newton", x_newton, "r", r)


### The data is saved to be loaded and analyzed in the file `plots_5-5_PentCaps_N370_Tersoff.jl`, where we generate plots and examine the results.