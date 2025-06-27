### Functions to generate the connectivity matrix and initial condition for all tubes

function get_5_5_connectivity_odd(numRings::Int=3)

    @assert isodd(numRings)

    connectivityCap_initial = [
        [2, 5, 6],
        [1, 7, 3],
        [2, 8, 4],
        [3, 9, 5],
        [4, 10, 1]
    ]

    for j in 6:10
        push!(connectivityCap_initial, [j - 5, 2 * j - 1, 2 * j])
    end

    for j in 11:2:19
        push!(connectivityCap_initial, [(j + 1) ÷ 2, j - 1 + 10 * (j == 11 ? 1 : 0), j + 10])
        push!(connectivityCap_initial, [(j + 1) ÷ 2, j + 2 - 10 * (j == 19 ? 1 : 0), j + 11])
    end

    connectivity = copy(connectivityCap_initial)

    for j in 1:numRings
        for k in 1:10
            currentIdx = length(connectivity) + 1  # MATLAB: size(connectivity,1) + 1
            third = currentIdx + 1 - 2 * mod(k + j, 2) +
                    10 * (k == 1 ? 1 : 0) * mod(j + 1, 2) -
                    10 * (k == 10 ? 1 : 0) * mod(j + 1, 2)
            push!(connectivity, [currentIdx + 10, currentIdx - 10, third])
        end
    end

    N = length(connectivity)  # should be 50 at this point

    closingCap = [[(N + 21) - val for val in row] for row in connectivityCap_initial]

    closingCap = reverse(closingCap)
    for row in closingCap
        push!(connectivity, row)
    end

    connectivityMat = hcat(connectivity...)'  # Convert to a 70×3 matrix

    nAtoms = size(connectivityMat, 1)  # 70 atoms
    x = zeros(Float64, nAtoms, 3)

    for i in 1:5
        x[i, 1] = cos(2π / 5 * i)
        x[i, 2] = sin(2π / 5 * i)
        x[i, 3] = 0.0
    end

    for i in 1:5
        x[5+i, 1] = cos(2π / 5 * i)
        x[5+i, 2] = sin(2π / 5 * i)
        x[5+i, 3] = 1.0
    end

    for j in 1:(numRings+2)
        for i in 1:10
            rowIdx = j * 10 + i  # j=1 gives rows 11:20, j=2 gives 21:30, etc.
            x[rowIdx, 1] = 2.5 * cos(2π / 10 * i)
            x[rowIdx, 2] = 2.5 * sin(2π / 10 * i)
            x[rowIdx, 3] = j + 1
        end
    end

    z_low = (numRings + 2) + 2
    for (idx, val) in enumerate(collect(5:-1:1))
        rowIdx = nAtoms - 9 + (idx - 1)
        x[rowIdx, 1] = cos(2π / 5 * val)
        x[rowIdx, 2] = sin(2π / 5 * val)
        x[rowIdx, 3] = z_low
    end
    z_high = (numRings + 2) + 3
    for (idx, val) in enumerate(collect(5:-1:1))
        rowIdx = nAtoms - 4 + (idx - 1)
        x[rowIdx, 1] = cos(2π / 5 * val)
        x[rowIdx, 2] = sin(2π / 5 * val)
        x[rowIdx, 3] = z_high
    end

    x .+= 0.01 * randn(nAtoms, 3)

    return connectivityMat, x
end


function get_5_5_connectivity_even(numRings)

    N = numRings * 10 + 60
    x = Array{Float64,2}(undef, N, 3)

    x[1:5, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) zeros(5)]
    x[6:10, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) ones(5)]

    for j = 1:numRings+4
        x[j*10+1:(j+1)*10, :] = [2.5 * cos.(2 * pi / 10 * (1:10)) 2.5 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
    end

    j = numRings + 4
    x[end-9:end-5, :] = [cos.(2 * pi / 5 * (5:-1:1)) sin.(2 * pi / 5 * (5:-1:1)) (j + 2) * ones(5)]
    x[end-4:end, :] = [cos.(2 * pi / 5 * (5:-1:1)) sin.(2 * pi / 5 * (5:-1:1)) (j + 3) * ones(5)]

    x = x + 0.01 * randn(size(x))

    ### Generate connectivity of nanotube with C60 caps
    connectivityCap = [2 5 6; 1 7 3; 2 8 4; 3 9 5; 4 10 1]

    for j = 6:10
        connectivityCap = [connectivityCap; [j - 5, 2 * j - 1, 2 * j]']
    end

    for j = 11:2:19
        connectivityCap = [connectivityCap; [(j + 1) ÷ 2, j - 1 + 10 * (j == 11), j + 10]']
        connectivityCap = [connectivityCap; [(j + 1) ÷ 2, j + 2 - 10 * (j == 19), j + 11]']
    end

    connectivity = connectivityCap

    ### Generate connectivity of rings within nanotube
    for j = 1:numRings+2
        for k = 1:10
            currentIdx = size(connectivity, 1) + 1
            connectivity = [connectivity; [currentIdx + 10, currentIdx - 10, currentIdx + 1 - 2 * mod(k + j, 2) + 10 * (k == 1) * mod(j + 1, 2) - 10 * (k == 10) * mod(j + 1, 2)]']
        end
    end

    ### Generate connectivity of closing cap(Top cap)
    if iseven(numRings)
        connectivityCapTopEven = [
            51 42 31
            55 41 32
            55 44 33
            54 43 34
            54 46 35
            53 45 36
            53 48 37
            52 47 38
            52 50 39
            51 49 40
            56 50 41
            57 48 49
            58 47 46
            59 44 45
            60 42 43
            57 51 60
            58 52 56
            59 53 57
            60 54 58
            59 56 55
        ]
        connectivity = [connectivity; connectivityCapTopEven .+ numRings * 10]
    else
        connectivityCap = reverse(size(connectivity, 1) + 21 .- connectivityCap, dims=1)
        connectivity = [connectivity; connectivityCap]
    end


    return connectivity, x
end

function get_connectivity_10_0_pentagonal(numRings)

    pentagonal_connectivity = [
        2 5 6;
        1 3 7;
        2 4 8;
        3 5 9;
        1 4 10;
        1 12 13;
        2 14 15;
        3 16 17;
        4 18 19;
        5 11 20;
        10 12 21;
        6 11 22;
        6 14 23;
        7 13 24;
        7 16 25;
        15 8 26;
        8 18 27;
        17 9 28;
        9 20 29;
        19 10 30;
        11 32 31;
        12 32 33;
        13 33 34;
        14 34 35;
        15 35 36;
        16 36 37;
        17 37 38;
        18 38 39;
        19 39 40;
        20 40 31;
        41 30 21;
        42 21 22;
        43 22 23;
        44 23 24;
        45 24 25;
        46 25 26;
        47 26 27;
        48 27 28;
        49 28 29;
        50 29 30;
        31 42 55;
        32 41 51;
        33 51 44;
        34 43 52;
        35 52 46;
        36 45 53;
        37 53 48;
        38 47 54;
        39 54 50;
        40 49 55;
        42 43 56;
        44 45 57;
        46 47 58;
        48 49 59;
        50 41 60;
        51 57 60;
        52 56 58;
        53 57 59;
        54 58 60;
        55 56 59
    ]

    connectivity = pentagonal_connectivity[1:40, :]
    cap = pentagonal_connectivity[end-19:end, :]


    for i = 1:numRings-1

        connectivity = [connectivity; connectivity[21:40, :] .+ i * 20]

    end

    connectivity = [connectivity; cap .+ (numRings - 1) * 20]

    N = 20 * numRings + 40
    x = Array{Float64,2}(undef, N, 3)
    x = zeros(N, 3)

    x[1:5, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) zeros(5)]
    x[6:10, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) 0.5 * ones(5)]

    k = 0
    for j = 0:2numRings+1
        if k == 0
            x[10+1+10*j:10+10*(j+1), :] = [2.5 * cos.(2 * pi / 10 * (1:10)) 2.5 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 1
        else
            x[10+1+10*j:10+10*(j+1), :] = [2 * cos.(2 * pi / 10 * (1:10)) 2 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 0
        end
    end

    j = 2numRings + 2
    x[end-9:end-5, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) (j + 1) * ones(5)]
    x[end-4:end, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) (j + 2) * ones(5)]


    return connectivity, x

end

function get_connectivity_10_0_hexagonal(numRings)


    hexagonal_connectivity =
        [
            2 6 7;
            1 3 8;
            2 4 9;
            3 5 10;
            4 6 11;
            5 1 12;
            1 15 14;
            2 16 15;
            3 17 18;
            4 20 19;
            5 21 20;
            6 13 22;
            12 14 23;
            7 13 24;
            7 8 25;
            17 8 26;
            16 9 27;
            9 19 28;
            18 10 29;
            10 11 30;
            11 22 31;
            12 21 32;
            13 33 34;
            14 34 35;
            15 35 36;
            16 36 37;
            17 37 38;
            18 38 39;
            19 39 40;
            20 40 41;
            21 41 42;
            22 42 33;
            32 23 43;
            23 24 44;
            24 25 45;
            25 26 46;
            26 27 47;
            27 28 48;
            28 29 49;
            29 30 50;
            30 31 51;
            31 32 52;
            33 44 58;
            34 43 53;
            35 53 54;
            36 47 54;
            46 37 55;
            38 49 55;
            39 48 56;
            40 56 57;
            41 52 57;
            42 51 58;
            44 45 59;
            45 46 60;
            47 48 61;
            49 50 62;
            50 51 63;
            52 43 64;
            53 60 64;
            54 59 61;
            55 60 62;
            61 63 56;
            57 62 64;
            58 59 63
        ]

    connectivity = hexagonal_connectivity[1:42, :]
    cap = hexagonal_connectivity[end-21:end, :]
    for i = 1:numRings-1

        connectivity = [connectivity; connectivity[23:42, :] .+ i * 20]

    end

    connectivity = [connectivity; cap .+ (numRings - 1) * 20]

    N = 20 * numRings + 44
    x = Array{Float64,2}(undef, N, 3)
    x = zeros(N, 3)

    x[1:6, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) zeros(6)]
    x[7:12, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) 0.5 * ones(6)]

    k = 0
    for j = 0:2numRings+1
        if k == 0
            x[12+1+10*j:12+10*(j+1), :] = [2 * cos.(2 * pi / 10 * (1:10)) 2 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 1
        else
            x[12+1+10*j:12+10*(j+1), :] = [2.5 * cos.(2 * pi / 10 * (1:10)) 2.5 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 0
        end
    end

    j = 2numRings + 2
    x[end-11:end-6, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) (j + 1) * ones(6)]
    x[end-5:end, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) (j + 2) * ones(6)]


    return connectivity, x

end

function get_connectivity_10_0_mixed(numRings)


    hexagonal_closing_cap = [
        33 44 58;
        34 43 53;
        35 53 54;
        36 47 54;
        46 37 55;
        38 49 55;
        39 48 56;
        40 56 57;
        41 52 57;
        42 51 58;
        44 45 59;
        45 46 60;
        47 48 61;
        49 50 62;
        50 51 63;
        52 43 64;
        53 60 64;
        54 59 61;
        55 60 62;
        61 63 56;
        57 62 64;
        58 59 63]

    mixed_connectivity = [
        2 5 6;
        1 3 7;
        2 4 8;
        3 5 9;
        1 4 10;
        1 12 13;
        2 14 15;
        3 16 17;
        4 18 19;
        5 11 20;
        10 12 21;
        6 11 22;
        6 14 23;
        7 13 24;
        7 16 25;
        15 8 26;
        8 18 27;
        17 9 28;
        9 20 29;
        19 10 30;
        11 32 31;
        12 32 33;
        13 33 34;
        14 34 35;
        15 35 36;
        16 36 37;
        17 37 38;
        18 38 39;
        19 39 40;
        20 40 31;
        41 30 21;
        42 21 22;
        43 22 23;
        44 23 24;
        45 24 25;
        46 25 26;
        47 26 27;
        48 27 28;
        49 28 29;
        50 29 30;
        hexagonal_closing_cap .- 2
    ]

    connectivity = mixed_connectivity[1:40, :]
    cap = mixed_connectivity[end-21:end, :]


    for i = 1:numRings-1

        connectivity = [connectivity; connectivity[21:40, :] .+ i * 20]

    end

    connectivity = [connectivity; cap .+ (numRings - 1) * 20]

    N = 20 * numRings + 42
    x = Array{Float64,2}(undef, N, 3)
    x = zeros(N, 3)

    x[1:5, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) zeros(5)]
    x[6:10, :] = [cos.(2 * pi / 5 * (1:5)) sin.(2 * pi / 5 * (1:5)) 0.5 * ones(5)]

    k = 0
    for j = 0:2numRings+1
        if k == 0
            x[10+1+10*j:10+10*(j+1), :] = [2.5 * cos.(2 * pi / 10 * (1:10)) 2.5 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 1
        else
            x[10+1+10*j:10+10*(j+1), :] = [2 * cos.(2 * pi / 10 * (1:10)) 2 * sin.(2 * pi / 10 * (1:10)) (j + 1) * ones(10)]
            k = 0
        end
    end

    j = 2numRings + 2
    x[end-11:end-6, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) (j + 1) * ones(6)]
    x[end-5:end, :] = [cos.(2 * pi / 6 * (1:6)) sin.(2 * pi / 6 * (1:6)) (j + 2) * ones(6)]


    return connectivity, x
end


