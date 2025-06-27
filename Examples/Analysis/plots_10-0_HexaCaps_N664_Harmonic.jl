using NanotubesCAPs
using IntervalArithmetic
using CairoMakie
using JLD2

using UnPack, Printf, LaTeXStrings, CairoMakie

### Load the data
data = load("data/10-0_HexaCaps_N664_Harmonic.jld2")
x_newton = data["x_newton"]
r = data["r"]

r = 1.22287e-11

x = reshape(x_newton[7:end], :, 3)
x = center_nanotube_zigzag(x)

x = interval.(x, r; format=:midpoint)

numRings = 31
connectivity, _ = get_connectivity_10_0_hexagonal(numRings)


### Energy
N = size(x, 1)
b = 1.44
θ = 2π / 3
kb = 469
kθ = 63
e = x -> harmonic_energy(x, connectivity, b, θ, kb, kθ)
E = e(x) / N
common_decimal_places([sup.(E); inf.(E)])
sup(E)
