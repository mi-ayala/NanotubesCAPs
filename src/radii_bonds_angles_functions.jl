### Some extra functions to compute radii, bonds and angles of nanotubes

function center_nanotube_armchair(x)

    x = reshape(x, :, 3)
    x = x .- mean(x, dims=1)

    v = vec(mean(x[1:5, :], dims=1) .- mean(x[end-4:end, :], dims=1))

    v = [0, 0, 1] - v / sqrt(dot(v, v))
    v = v / sqrt(dot(v, v))
    x = x .- 2 * (x * v) * v'

    return x

end

function center_nanotube_zigzag(x)

    x = reshape(x, :, 3)
    x = x .- mean(x, dims=1)

    v = vec(mean(x[1:6, :], dims=1) .- mean(x[end-5:end, :], dims=1))

    v = [0, 0, 1] - v / sqrt(dot(v, v))
    v = v / sqrt(dot(v, v))
    x = x .- 2 * (x * v) * v'

    return x

end

function get_ring_indices(ring_size, boundary_size, N)

    number_of_rings = Int((N - 2 * boundary_size) / ring_size)
    ring_indices = zeros(Int32, number_of_rings, ring_size)

    for i in 1:number_of_rings
        ring_indices[i, :] = (boundary_size+1+(i-1)*ring_size):(boundary_size+i*ring_size)
    end

    return ring_indices

end

function get_radii(x)

    center = sum(x, dims=1) / interval(10)

    r = x .- center

    # println("circle center: ", center)

    sqrt.(sum(r .^ 2, dims=2))
end

function radii_plot(
    x1, range, whisker; output_path="my_figure.pdf"
)

    ### Color pallete
    colors = [
        # "#000000",  # black
        # "#999999",  # grey
        "#0072B2",  # deep blue
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#CC79A7",  # muted purple
        "#7570B3",  # soft purple
        "#E69F00",  # gold/orange
        "#D55E00",  # orange-red
        "#E1E171",   # softened yellow (was F0E442)
        "#8B008B",  # dark magenta (new)
        "#A0522D",  # warm brown   (new)
    ]

    ### Determine ring indices
    N = size(x1, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)
    upper_bound = zeros(Float64, number_of_rings)
    lower_bound = zeros(Float64, number_of_rings)

    half_rings = Int((number_of_rings - 1) / 2)



    ### Add padding in y axis
    for i in 1:number_of_rings

        if i ∉ range
            continue
        end

        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]

        ring_intervals = abs.(get_radii(ring_coords))
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)

        upper_bound[i] = maximum(sup.(ring_intervals))
        lower_bound[i] = minimum(inf.(ring_intervals))

    end

    fig = Figure(
        size=(2000, 700),
        dpi=600,
        backgroundcolor=:white
    )

    ax = Axis(
        fig[1, 1],
        xlabel=L"\text{Index of cross section}",
        ylabel=L"\text{Distance to tube's axis [\AA]}",
        titlesize=18,
        #  yscale=log10,
        xlabelsize=50,
        ylabelsize=50,
        xticklabelsize=45,
        yticklabelsize=45,
        topspinevisible=true,
        rightspinevisible=true,
        xgridvisible=false,
        ygridvisible=false,
        ytickformat=ys -> [@sprintf("%.10f", y) for y in ys],
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

    ###  Loop over each ring
    for i in 1:number_of_rings

        if i ∉ range
            continue
        end
        ring_idxs = rings_indices[i, :]
        ring_coords = x1[ring_idxs, :]

        ring_intervals = abs.(get_radii(ring_coords))
        ring_intervals = sort(ring_intervals, dims=1, by=sup, rev=true)


        for j in 1:10
            scatter!(ax, [i - half_rings - 1], [mid.(ring_intervals[j])], color=colors[1], markersize=30)
            rangebars!(
                ax,
                [i - half_rings - 1],
                [inf.(ring_intervals[j])],
                [sup.(ring_intervals[j])],
                color=colors[1],
                linewidth=1,
                whiskerwidth=whisker,
                alpha=0.7,
                transparency=true
            )
        end

    end


    save(output_path, fig)
    fig
end

function sort_armchair_bonds(x, connectivity)
    ### Ruturns the indices of horizontal and rising bonds for each ring. Tube is assumed to be aligend to the z-axis.

    N = size(x, 1)
    ring_count = 10
    boundary_size = 30
    rings_indices = get_ring_indices(ring_count, boundary_size, N)
    number_of_rings = size(rings_indices, 1)

    horizontal_bonds_connect = []
    rising_bonds_connect = []

    for i in 1:number_of_rings

        for j in 1:10

            x_neighbours = [connectivity[rings_indices[i, j], :] x[connectivity[rings_indices[i, j], :], :]]
            ordered_neighbours = sortslices(x_neighbours, dims=1, by=x -> x[4])

            ### Horizontal bond
            bound_h = Int.([rings_indices[i, j] ordered_neighbours[2, 1]])

            ### Rising bonds
            bound_r = Int.([rings_indices[i, j] ordered_neighbours[3, 1]])

            if !in([bound_h[2] bound_h[1]], horizontal_bonds_connect)
                push!(horizontal_bonds_connect, bound_h)
            end

            if !in([bound_r[2] bound_r[1]], rising_bonds_connect)
                push!(rising_bonds_connect, bound_r)
            end


        end


    end

    return horizontal_bonds_connect, rising_bonds_connect

end

