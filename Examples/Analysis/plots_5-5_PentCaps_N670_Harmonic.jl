using NanotubesCAPs
using IntervalArithmetic
using Statistics
using CairoMakie
using JLD2

using UnPack, Printf, LaTeXStrings


### Load the data
data = load("data/5-5_PentCaps_N670_Harmonic.jld2")
x_newton = data["x_newton"]
r = data["r"]

r = 4.68436e-11
x = reshape(x_newton[7:end], :, 3)

x = center_nanotube_armchair(x)

x = interval.(x, r; format=:midpoint)

connectivity, _ = get_5_5_connectivity_odd(63)


# x = reshape(x, :, 3)


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


radii = sqrt.(x[:, 1] .^ 2 + x[:, 2] .^ 2)
sup(radii[331:340][1])
common_decimal_places([sup.(radii[331:340]); inf.(radii[331:340])])
reference_radius = mean(radii[331:340])


function radii_plot_signed_deviation(
    x1, range, reference_radius, whisker;
    output_path="my_figure.pdf", scale_tol=1e-8
)

    # --- Custom log-linear transformation ---
    function smooth_symlog(y; t=scale_tol)
        sign.(y) .* ifelse.(abs.(y) .< t, abs.(y) / t, log10.(abs.(y) ./ t) .+ 1)
    end

    # --- Color palette ---
    colors = [
        "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#7570B3",
        "#E69F00", "#D55E00", "#E1E171", "#8B008B", "#A0522D",
    ]

    # --- Prepare ring data ---
    N = size(x1, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)

    upper_bound = zeros(Float64, number_of_rings)
    lower_bound = zeros(Float64, number_of_rings)
    mid_bound = zeros(Float64, number_of_rings)

    for i in 1:number_of_rings
        if i ∉ range
            continue
        end
        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]
        ring_intervals = get_radii(ring_coords) .- interval(reference_radius)
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        upper_bound[i] = maximum(sup.(ring_intervals))
        lower_bound[i] = minimum(inf.(ring_intervals))
        mid_bound[i] = minimum(mid.(ring_intervals))
    end

    # --- Set up figure and axis ---
    fig = Figure(size=(2000, 600), dpi=600, backgroundcolor=:white)
    half_rings = Int((number_of_rings - 1) / 2)

    # Define physical tick values and their transformed positions
    tick_physical = [-1e-4, -1e-6, -1e-8, 0.0, 1e-8, 1e-6, 1e-4]
    tick_positions = smooth_symlog(tick_physical)

    tick_labels = [
        L"-10^{-4}",
        L"-10^{-6}",
        L"-10^{-8}",
        L"0",
        L"10^{-8}",
        L"10^{-6}",
        L"10^{-4}"
    ]



    ax = Axis(
        fig[1, 1],
        # xlabel=L"\text{Index of cross section}",
        # ylabel=L"\Delta r~\text{[\AA]}",
        titlesize=18,
        xlabelsize=50,
        ylabelsize=50,
        xticklabelsize=45,
        yticklabelsize=45,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        yticks=(tick_positions, tick_labels),
        xtickalign=1,
        ytickalign=1,
        yticksmirrored=true,
        xticksmirrored=true,
        xticksize=15,
        yticksize=15,
        spinewidth=4,
        xtickwidth=4,
        ytickwidth=4
    )

    # --- Plot each ring's deviations ---
    for i in 1:number_of_rings
        if i ∉ range
            continue
        end

        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]
        ring_intervals = get_radii(ring_coords) .- interval(reference_radius)
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        for j in 1:10
            x = i - half_rings - 1
            y_mid = smooth_symlog(mid.(ring_intervals[j]))
            y_lo = smooth_symlog(inf.(ring_intervals[j]))
            y_hi = smooth_symlog(sup.(ring_intervals[j]))

            scatter!(ax, [x], [y_mid], color=colors[1], markersize=10)
            rangebars!(
                ax,
                [x],
                [y_lo],
                [y_hi],
                color=colors[1],
                linewidth=5,
                whiskerwidth=whisker,
                alpha=0.9,
                transparency=true
            )
        end
    end

    # --- Add smoothed midline ---
    xvals = -half_rings:half_rings
    mid_transformed = smooth_symlog.(mid.(mid_bound))
    lines!(ax, xvals, mid_transformed, color="gray70", linewidth=2)

    save(output_path, fig)
    return fig
end

whisker = 0.0
radii_plot_signed_deviation(x, 1:61, reference_radius, whisker; output_path="signed_difference_radius_5-5_PentCaps_N670_Harmonic.pdf", scale_tol=1e-9)

whisker = 30
radii_plot(x, 14:48, whisker; output_path="radius_5-5_PentCaps_N670_Harmonic.pdf")


