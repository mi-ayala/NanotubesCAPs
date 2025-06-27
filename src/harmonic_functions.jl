### Harmonic potential functions

function harmonic_energy(x_input, connectivity, b, θ, kb, kθ)

    x = transpose(reshape(x_input, :, 3))
    N = size(x, 2)

    e1 = zero(typeof(x_input[1]))
    e2 = zero(typeof(x_input[1]))

    for i in 1:N
        ### Computing neighbors        
        y = view(x, :, connectivity[i, :])

        for k = 1:3
            v = view(y, :, k) - view(x, :, i)
            dist = sqrt(dot(v, v))
            e1 += kb * (dist - b)^2

            idx1 = k
            idx2 = mod(k, 3) + 1
            v1 = view(y, :, idx1) - view(x, :, i)
            v2 = view(y, :, idx2) - view(x, :, i)
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm

            angle = acos(proj)
            e2 += kθ * (angle - θ)^2
        end
    end

    return 0.5 * e1 + e2

end

function harmonic_energy(x_input, connectivity, b::IntervalArithmetic.Interval, θ::IntervalArithmetic.Interval, kb::IntervalArithmetic.Interval, kθ::IntervalArithmetic.Interval)

    x = transpose(reshape(x_input, :, 3))
    N = size(x, 2)

    e1 = zero(typeof(x_input[1]))
    e2 = zero(typeof(x_input[1]))

    for i in 1:N
        ### Computing neighbors        
        y = view(x, :, connectivity[i, :])

        for k = 1:3
            v = view(y, :, k) - view(x, :, i)
            dist = sqrt(dot(v, v))
            e1 += kb * (dist - b)^2

            idx1 = k
            idx2 = mod(k, 3) + 1
            v1 = view(y, :, idx1) - view(x, :, i)
            v2 = view(y, :, idx2) - view(x, :, i)
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm

            angle = acos(proj)
            e2 += kθ * (angle - θ)^2
        end
    end

    return interval(0.5) * e1 + e2

end

function Grad(x, connectivity, b, θ, kb, kθ)
    x = reshape(x, :, 3)
    n = size(connectivity, 1)
    e = zero(eltype(x[1]))
    de = zeros(eltype(x[1]), n, 3)

    ### Potential
    vBond = d -> kb * (d - b)^2
    vBondPrime = d -> 2 * kb * (d - b)

    vAngle = a -> kθ * (a - θ)^2
    vAnglePrime = a -> 2 * kθ * (a - θ)

    ### Pairwise interactions
    for j = 1:n
        for k = 1:3
            v = x[connectivity[j, k], :] - x[j, :]
            dist = sqrt(dot(v, v))
            e += vBond(dist)

            de[j, :] .+= (vBondPrime(dist) / dist) .* (-v)
            de[connectivity[j, k], :] .+= (vBondPrime(dist) / dist) .* v
        end
    end

    e = 0.5 * e
    de = 0.5 * de
    ### Angle interactions
    for j = 1:n
        for k = 1:3
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm
            angle = acos(proj)
            e += vAngle(angle)


            dv1 = v2 / nrm - (proj / dot(v1, v1)) * v1
            dv2 = v1 / nrm - (proj / dot(v2, v2)) * v2

            de[j, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) * (-dv1 - dv2)
            de[idx1, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv1
            de[idx2, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv2

        end
    end

    de = de[:]

    return e, de
end

function Grad(x, connectivity, b::IntervalArithmetic.Interval, θ::IntervalArithmetic.Interval, kb::IntervalArithmetic.Interval, kθ::IntervalArithmetic.Interval)

    x = reshape(x, :, 3)
    n = size(connectivity, 1)
    e = zero(eltype(x[1]))
    de = zeros(eltype(x[1]), n, 3)

    ### Potential
    vBond = d -> kb * (d - b)^2
    vBondPrime = d -> interval(2) * kb * (d - b)

    vAngle = a -> kθ * (a - θ)^2
    vAnglePrime = a -> interval(2) * kθ * (a - θ)

    ### Pairwise interactions
    for j = 1:n
        for k = 1:3
            v = x[connectivity[j, k], :] - x[j, :]
            dist = sqrt(dot(v, v))
            e += vBond(dist)

            de[j, :] .+= (vBondPrime(dist) / dist) .* (-v)
            de[connectivity[j, k], :] .+= (vBondPrime(dist) / dist) .* v
        end
    end

    e = interval(0.5) * e
    de = interval(0.5) * de
    ### Angle interactions
    for j = 1:n
        for k = 1:3
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm
            angle = acos(proj)
            e += vAngle(angle)


            dv1 = v2 / nrm - (proj / dot(v1, v1)) * v1
            dv2 = v1 / nrm - (proj / dot(v2, v2)) * v2

            de[j, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) * (-dv1 - dv2)
            de[idx1, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv1
            de[idx2, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv2

        end
    end

    de = de[:]

    return e, de
end

function extended_Grad(x_input, x_fix, connectivity, b, θ, kb, kθ)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing vector field
    _, Fx = Energy_Grad(x, connectivity, b, θ, kb, kθ)

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]

    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x

    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]


end

function extended_Grad(x_input, x_fix, connectivity, b::IntervalArithmetic.Interval, θ::IntervalArithmetic.Interval, kb::IntervalArithmetic.Interval, kθ::IntervalArithmetic.Interval)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing vector field
    _, Fx = Energy_Grad(x, connectivity, b, θ, kb, kθ)

    ### Extending function
    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]

    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + interval(mu1) * T1 + interval(mu2) * T2 + interval(mu3) * T3 + interval(lambda1) * I1x + interval(lambda2) * I2x + interval(lambda3) * I3x

    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]


end

function extended_Hess(x_input, x_fix, connectivity, b, θ, kb, kθ)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing Fx
    ### Computing vector field
    _, Fx = Energy_Grad(reshape(x, :, 3), connectivity, b, θ, kb, kθ)

    ### Extending function

    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]

    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]


    ### Second alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1x + lambda2 * I2x + lambda3 * I3x


    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]

    ### Computing DFx
    vBondPrime = d -> 2 * kb * (d - b)
    vBondPrimePrime = d -> 2 * kb

    x = reshape(x, :, 3)

    ### Prelocating Matrices

    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    for j = 1:N

        for k = 1:3

            i = connectivity[j, k]
            v = x[i, :] - x[j, :]
            dist = sqrt(dot(v, v))

            ### With respect to x
            dxj = (v[1] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist) .* v
            dxi = (v[1] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist) .* v

            Dxx[j, j] = Dxx[j, j] - dxj[1] + vBondPrime(dist) / dist
            Dxx[i, i] = Dxx[i, i] + dxi[1] + vBondPrime(dist) / dist

            Dxy[j, j] = Dxy[j, j] - dxj[2]
            Dxy[i, i] = Dxy[i, i] + dxi[2]
            Dxz[j, j] = Dxz[j, j] - dxj[3]
            Dxz[i, i] = Dxz[i, i] + dxi[3]

            ### Mixed derivatives
            Dxx[j, i] = Dxx[j, i] - (dxi[1] + vBondPrime(dist) / dist)
            Dxx[i, j] = Dxx[i, j] + (dxj[1] - vBondPrime(dist) / dist)
            Dxy[j, i] = Dxy[j, i] - dxi[2]
            Dxy[i, j] = Dxy[i, j] + dxj[2]
            Dxz[j, i] = Dxz[j, i] - dxi[3]
            Dxz[i, j] = Dxz[i, j] + dxj[3]

            ###  With respect to y
            dyj = v .* (v[2] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dyi = v .* (v[2] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dyy[j, j] = Dyy[j, j] - dyj[2] + vBondPrime(dist) / dist
            Dyy[i, i] = Dyy[i, i] + dyi[2] + vBondPrime(dist) / dist
            Dyz[j, j] = Dyz[j, j] - dyj[3]
            Dyz[i, i] = Dyz[i, i] + dyi[3]

            ### Mixed derivatives
            Dyy[j, i] = Dyy[j, i] - (dyi[2] + vBondPrime(dist) / dist)
            Dyy[i, j] = Dyy[i, j] + (dyj[2] - vBondPrime(dist) / dist)
            Dyz[j, i] = Dyz[j, i] - dyi[3]
            Dyz[i, j] = Dyz[i, j] + dyj[3]

            ### with respect to z
            dzj = v .* (v[3] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dzi = v .* (v[3] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dzz[j, j] = Dzz[j, j] - dzj[3] + vBondPrime(dist) / dist
            Dzz[i, i] = Dzz[i, i] + dzi[3] + vBondPrime(dist) / dist

            ### Mixed derivatives
            Dzz[j, i] = Dzz[j, i] - (dzi[3] + vBondPrime(dist) / dist)
            Dzz[i, j] = Dzz[i, j] + (dzj[3] - vBondPrime(dist) / dist)

        end

    end


    DFx = 0.5 * [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]

    ### Prelocating Matrices
    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)

    Dyx = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzx = zeros(type, N, N)
    Dzy = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    ### ANGLE INTERACTION HESSIAN 
    vAnglePrime = a -> 2 * kθ * (a - θ)
    vAnglePrimePrime = a -> 2 * kθ


    for j = 1:N

        for k = 1:3

            ### Neighbour atoms 
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]

            ### These are vi and vk
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]

            ### These are vi and vk
            dv1_0 = [-1, -1, -1]
            dv1_1 = [1, 1, 1]
            dv1_2 = [0, 0, 0]

            dv2_0 = [-1, -1, -1]
            dv2_1 = [0, 0, 0]
            dv2_2 = [1, 1, 1]

            ### Norms squared
            v1v1 = dot(v1, v1)
            v2v2 = dot(v2, v2)

            ### Derivatives of v1v1 v2v2
            dv1v1_0 = -2 * v1
            dv1v1_1 = 2 * v1
            dv1v1_2 = [0, 0, 0]

            dv2v2_0 = -2 * v2
            dv2v2_1 = [0, 0, 0]
            dv2v2_2 = 2 * v2

            ### Norm nrm
            nrm = sqrt(v1v1 * v2v2)

            ### Derivatives of nrm (dnrm_0 = dnrm1 + dnrm2)        
            dnrm_1 = (v2v2 / nrm) * v1
            dnrm_2 = (v1v1 / nrm) * v2
            dnrm_0 = -dnrm_1 - dnrm_2

            ### Angle and proj
            proj = dot(v1, v2) / nrm
            rad = sqrt(1.0 - proj^2)
            rad2 = 1.0 - proj^2

            angle = acos(proj)

            ### Derivatives of proj(dproj_0 = -dv1-dv2 )
            dproj_1 = v2 / nrm - (proj / v1v1) * v1
            dproj_2 = v1 / nrm - (proj / v2v2) * v2
            dproj_0 = -dproj_1 .- dproj_2

            ### u1 and u2
            u1 = v1 / v1v1
            u2 = v2 / v2v2

            ### Derivatives of u1 and u2
            du1_0 = dv1_0 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_0
            du1_1 = dv1_1 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_1
            du1_2 = dv1_2 / v1v1 - (1 / (v1v1^2)) * v1 .* dv1v1_2
            du2_0 = dv2_0 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_0
            du2_1 = dv2_1 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_1
            du2_2 = dv2_2 / v2v2 - (1 / (v2v2^2)) * v2 .* dv2v2_2


            a = [1, 1, 2, 2, 3, 3]
            b = [2, 3, 3, 1, 1, 2]

            ### BUILDING DERIVATIVES FOR dv1 
            ## dv1 wrt Xi  
            dproj_11 = dv2_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v2 - proj .* du1_1 - u1 .* dproj_1
            Dv_11 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_1) / rad2 + dproj_11) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_1) / rad2

            Dxx[idx1, idx1] = Dxx[idx1, idx1] + Dv_11[1]
            Dyy[idx1, idx1] = Dyy[idx1, idx1] + Dv_11[2]
            Dzz[idx1, idx1] = Dzz[idx1, idx1] + Dv_11[3]

            dproj_11mixed = -dnrm_1[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_1[b] * (v1v1^-2)) - u1[a] .* dproj_1[b]
            Dv_11mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_1[b] / rad2) + dproj_11mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_1[b] / rad2)

            Dxy[idx1, idx1] = Dxy[idx1, idx1] + Dv_11mixed[1]
            Dxz[idx1, idx1] = Dxz[idx1, idx1] + Dv_11mixed[2]
            Dyz[idx1, idx1] = Dyz[idx1, idx1] + Dv_11mixed[3]
            Dyx[idx1, idx1] = Dyx[idx1, idx1] + Dv_11mixed[4]
            Dzx[idx1, idx1] = Dzx[idx1, idx1] + Dv_11mixed[5]
            Dzy[idx1, idx1] = Dzy[idx1, idx1] + Dv_11mixed[6]

            ### dv1 wrt Xk
            dproj_12 = dv2_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v2 - proj .* du1_2 - u1 .* dproj_2
            Dv_12 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_2) / rad2 + dproj_12) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_2) / rad2

            Dxx[idx1, idx2] = Dxx[idx1, idx2] + Dv_12[1]
            Dyy[idx1, idx2] = Dyy[idx1, idx2] + Dv_12[2]
            Dzz[idx1, idx2] = Dzz[idx1, idx2] + Dv_12[3]

            dproj_12mixed = -dnrm_2[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_2[b] * (v1v1^-2)) - u1[a] .* dproj_2[b]
            Dv_12mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_2[b] / rad2) + dproj_12mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_2[b] / rad2)

            Dxy[idx1, idx2] = Dxy[idx1, idx2] + Dv_12mixed[1]
            Dxz[idx1, idx2] = Dxz[idx1, idx2] + Dv_12mixed[2]
            Dyz[idx1, idx2] = Dyz[idx1, idx2] + Dv_12mixed[3]
            Dyx[idx1, idx2] = Dyx[idx1, idx2] + Dv_12mixed[4]
            Dzx[idx1, idx2] = Dzx[idx1, idx2] + Dv_12mixed[5]
            Dzy[idx1, idx2] = Dzy[idx1, idx2] + Dv_12mixed[6]

            ### dv1 wrt Xj   
            dproj_10 = dv2_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v2 - proj .* du1_0 - u1 .* dproj_0
            Dv_10 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_10) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[idx1, j] = Dxx[idx1, j] + Dv_10[1]
            Dyy[idx1, j] = Dyy[idx1, j] + Dv_10[2]
            Dzz[idx1, j] = Dzz[idx1, j] + Dv_10[3]

            dproj_10mixed = -dnrm_0[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_0[b] * (v1v1^-2)) - u1[a] .* dproj_0[b]
            Dv_10mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_0[b] / rad2) + dproj_10mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_0[b] / rad2)

            Dxy[idx1, j] = Dxy[idx1, j] + Dv_10mixed[1]
            Dxz[idx1, j] = Dxz[idx1, j] + Dv_10mixed[2]
            Dyz[idx1, j] = Dyz[idx1, j] + Dv_10mixed[3]
            Dyx[idx1, j] = Dyx[idx1, j] + Dv_10mixed[4]
            Dzx[idx1, j] = Dzx[idx1, j] + Dv_10mixed[5]
            Dzy[idx1, j] = Dzy[idx1, j] + Dv_10mixed[6]

            ### BUILDING DERIVATIVES FOR dv2 
            ### dv2 wrt Xi
            dproj_21 = dv1_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v1 - proj .* du2_1 - u2 .* dproj_1
            Dv_21 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_1) / rad2 + dproj_21) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_1) / rad2

            Dxx[idx2, idx1] = Dxx[idx2, idx1] + Dv_21[1]
            Dyy[idx2, idx1] = Dyy[idx2, idx1] + Dv_21[2]
            Dzz[idx2, idx1] = Dzz[idx2, idx1] + Dv_21[3]

            dproj_21mixed = -dnrm_1[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_1[b] * (v2v2^-2)) - u2[a] .* dproj_1[b]
            Dv_21mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_1[b] / rad2) + dproj_21mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_1[b] / rad2)

            Dxy[idx2, idx1] = Dxy[idx2, idx1] + Dv_21mixed[1]
            Dxz[idx2, idx1] = Dxz[idx2, idx1] + Dv_21mixed[2]
            Dyz[idx2, idx1] = Dyz[idx2, idx1] + Dv_21mixed[3]
            Dyx[idx2, idx1] = Dyx[idx2, idx1] + Dv_21mixed[4]
            Dzx[idx2, idx1] = Dzx[idx2, idx1] + Dv_21mixed[5]
            Dzy[idx2, idx1] = Dzy[idx2, idx1] + Dv_21mixed[6]

            ### dv2 wrt Xk
            dproj_22 = dv1_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v1 - proj .* du2_2 - u2 .* dproj_2
            Dv_22 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_2) / rad2 + dproj_22) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_2) / rad2

            Dxx[idx2, idx2] = Dxx[idx2, idx2] + Dv_22[1]
            Dyy[idx2, idx2] = Dyy[idx2, idx2] + Dv_22[2]
            Dzz[idx2, idx2] = Dzz[idx2, idx2] + Dv_22[3]

            dproj_22mixed = -dnrm_2[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_2[b] * (v2v2^-2)) - u2[a] .* dproj_2[b]
            Dv_22mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_2[b] / rad2) + dproj_22mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_2[b] / rad2)

            Dxy[idx2, idx2] = Dxy[idx2, idx2] + Dv_22mixed[1]
            Dxz[idx2, idx2] = Dxz[idx2, idx2] + Dv_22mixed[2]
            Dyz[idx2, idx2] = Dyz[idx2, idx2] + Dv_22mixed[3]
            Dyx[idx2, idx2] = Dyx[idx2, idx2] + Dv_22mixed[4]
            Dzx[idx2, idx2] = Dzx[idx2, idx2] + Dv_22mixed[5]
            Dzy[idx2, idx2] = Dzy[idx2, idx2] + Dv_22mixed[6]

            ### dv2 wrt Xj  
            dproj_20 = dv1_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v1 - proj .* du2_0 - u2 .* dproj_0
            Dv_20 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_20) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2

            Dxx[idx2, j] = Dxx[idx2, j] + Dv_20[1]
            Dyy[idx2, j] = Dyy[idx2, j] + Dv_20[2]
            Dzz[idx2, j] = Dzz[idx2, j] + Dv_20[3]

            dproj_20mixed = -dnrm_0[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_0[b] * (v2v2^-2)) - u2[a] .* dproj_0[b]
            Dv_20mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_0[b] / rad2) + dproj_20mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_0[b] / rad2)

            Dxy[idx2, j] = Dxy[idx2, j] + Dv_20mixed[1]
            Dxz[idx2, j] = Dxz[idx2, j] + Dv_20mixed[2]
            Dyz[idx2, j] = Dyz[idx2, j] + Dv_20mixed[3]
            Dyx[idx2, j] = Dyx[idx2, j] + Dv_20mixed[4]
            Dzx[idx2, j] = Dzx[idx2, j] + Dv_20mixed[5]
            Dzy[idx2, j] = Dzy[idx2, j] + Dv_20mixed[6]


            ### BUILDING DERIVATIVES FOR dv0 
            ### dv0 wrt Xi

            dproj_01 = -dproj_11 - dproj_21
            Dv_01 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_01) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[j, idx1] = Dxx[j, idx1] + Dv_01[1]
            Dyy[j, idx1] = Dyy[j, idx1] + Dv_01[2]
            Dzz[j, idx1] = Dzz[j, idx1] + Dv_01[3]
            Dxy[j, idx1] = Dxy[j, idx1] - Dv_11mixed[1] - Dv_21mixed[1]
            Dxz[j, idx1] = Dxz[j, idx1] - Dv_11mixed[2] - Dv_21mixed[2]
            Dyz[j, idx1] = Dyz[j, idx1] - Dv_11mixed[3] - Dv_21mixed[3]
            Dyx[j, idx1] = Dyx[j, idx1] - Dv_11mixed[4] - Dv_21mixed[4]
            Dzx[j, idx1] = Dzx[j, idx1] - Dv_11mixed[5] - Dv_21mixed[5]
            Dzy[j, idx1] = Dzy[j, idx1] - Dv_11mixed[6] - Dv_21mixed[6]



            ### dv0 wrt Xk
            dproj_02 = -dproj_12 - dproj_22
            Dv_02 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_02) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2


            Dxx[j, idx2] = Dxx[j, idx2] + Dv_02[1]
            Dyy[j, idx2] = Dyy[j, idx2] + Dv_02[2]
            Dzz[j, idx2] = Dzz[j, idx2] + Dv_02[3]
            Dxy[j, idx2] = Dxy[j, idx2] - Dv_12mixed[1] - Dv_22mixed[1]
            Dxz[j, idx2] = Dxz[j, idx2] - Dv_12mixed[2] - Dv_22mixed[2]
            Dyz[j, idx2] = Dyz[j, idx2] - Dv_12mixed[3] - Dv_22mixed[3]
            Dyx[j, idx2] = Dyx[j, idx2] - Dv_12mixed[4] - Dv_22mixed[4]
            Dzx[j, idx2] = Dzx[j, idx2] - Dv_12mixed[5] - Dv_22mixed[5]
            Dzy[j, idx2] = Dzy[j, idx2] - Dv_12mixed[6] - Dv_22mixed[6]

            ### dv0 wrt Xj      
            dproj_00 = -dproj_10 - dproj_20
            Dv_00 = -(vAnglePrime(angle) / rad) * (proj * (dproj_0 .* dproj_0) / rad2 + dproj_00) + vAnglePrimePrime(angle) * (dproj_0 .* dproj_0) / rad2

            Dxx[j, j] += Dv_00[1]
            Dyy[j, j] += Dv_00[2]
            Dzz[j, j] += Dv_00[3]
            Dxy[j, j] += -Dv_10mixed[1] - Dv_20mixed[1]
            Dxz[j, j] += -Dv_10mixed[2] - Dv_20mixed[2]
            Dyz[j, j] += -Dv_10mixed[3] - Dv_20mixed[3]
            Dyx[j, j] += -Dv_10mixed[4] - Dv_20mixed[4]
            Dzx[j, j] += -Dv_10mixed[5] - Dv_20mixed[5]
            Dzy[j, j] += -Dv_10mixed[6] - Dv_20mixed[6]

        end

    end

    dFx = DFx + [Dxx Dxy Dxz; Dyx Dyy Dyz; Dzx Dzy Dzz]


    ### Extending derivative
    dFx = [zeros(type, 3 * N, 6) dFx]
    dFx = [zeros(type, 6, 3 * N + 6); dFx]

    Adjusts = [zeros(type, N, N) -lambda1*Diagonal(ones(type, N)) lambda3*Diagonal(ones(type, N)); lambda1*Diagonal(ones(type, N)) zeros(type, N, N) -lambda2*Diagonal(ones(type, N)); -lambda3*Diagonal(ones(type, N)) lambda2*Diagonal(ones(type, N)) zeros(type, N, N)]
    Adjusts = [I1x I2x I3x T1 T2 T3 Adjusts]
    Adjusts = [zeros(type, 1, 6) transpose(I1xfix); zeros(type, 1, 6) transpose(I2xfix); zeros(type, 1, 6) transpose(I3xfix); zeros(type, 1, 6) transpose(T1); zeros(type, 1, 6) transpose(T2); zeros(type, 1, 6) transpose(T3); Adjusts]

    dFx = dFx + Adjusts

    return dFx

end

function extended_Hess(x_input, x_fix, connectivity, b::IntervalArithmetic.Interval, θ::IntervalArithmetic.Interval, kb::IntervalArithmetic.Interval, kθ::IntervalArithmetic.Interval)

    ### Input handling
    x_fix = reshape(x_fix, :, 1)
    x_input = reshape(x_input, 1, :)
    N = size(connectivity, 1)
    type = typeof(x_input[1])

    ### Rotations unfolding parameters
    lambda1 = x_input[1]
    lambda2 = x_input[2]
    lambda3 = x_input[3]

    ### Translations unfolding parameters
    mu1 = x_input[4]
    mu2 = x_input[5]
    mu3 = x_input[6]

    x = reshape(x_input[7:end], :, 1)

    ### Computing Fx
    ### Computing vector field
    _, Fx = Energy_Grad(reshape(x, :, 3), connectivity, b, θ, kb, kθ)

    ### Extending function

    ### Translation generators 
    T1 = [ones(type, N, 1); zeros(type, 2 * N, 1)]
    T2 = [zeros(type, N, 1); ones(type, N, 1); zeros(type, N, 1)]
    T3 = [zeros(type, 2 * N, 1); ones(type, N, 1)]

    T1x = [ones(type, 1, N) zeros(type, 1, 2 * N)] * x
    T2x = [zeros(type, 1, N) ones(type, 1, N) zeros(type, 1, N)] * x
    T3x = [zeros(type, 1, 2 * N) ones(type, 1, N)] * x

    ### Rotations generators
    I1xfix = [-x_fix[N+1:2N, :]; x_fix[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2xfix = [zeros(type, N, size(x_fix, 2)); -x_fix[2N+1:end, :]; x_fix[N+1:2N, :]]
    I3xfix = [x_fix[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x_fix[1:N, :]]


    ### Rotations generators at x
    I1x = [-x[N+1:2N, :]; x[1:N, :]; zeros(type, N, size(x_fix, 2))]
    I2x = [zeros(type, N, size(x_fix, 2)); -x[2N+1:end, :]; x[N+1:2N, :]]
    I3x = [x[2N+1:end, :]; zeros(type, N, size(x_fix, 2)); -x[1:N, :]]


    ### First alternative
    Fx = reshape(Fx, 3 * N, 1) + mu1 * T1 + mu2 * T2 + mu3 * T3 + lambda1 * I1xfix + lambda2 * I2xfix + lambda3 * I3xfix


    ### Balancing equations   
    Fx = [x' * (I1xfix); x' * (I2xfix); x' * (I3xfix); T1x; T2x; T3x; Fx]

    ### Computing DFx
    vBondPrime = d -> interval(2) * kb * (d - b)
    vBondPrimePrime = d -> interval(2) * kb

    x = reshape(x, :, 3)

    ### Prelocating Matrices

    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    for j = 1:N

        for k = 1:3

            i = connectivity[j, k]
            v = x[i, :] - x[j, :]
            dist = sqrt(dot(v, v))

            ### With respect to x
            dxj = (v[1] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist) .* v
            dxi = (v[1] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist) .* v

            Dxx[j, j] = Dxx[j, j] - dxj[1] + vBondPrime(dist) / dist
            Dxx[i, i] = Dxx[i, i] + dxi[1] + vBondPrime(dist) / dist

            Dxy[j, j] = Dxy[j, j] - dxj[2]
            Dxy[i, i] = Dxy[i, i] + dxi[2]
            Dxz[j, j] = Dxz[j, j] - dxj[3]
            Dxz[i, i] = Dxz[i, i] + dxi[3]

            ### Mixed derivatives
            Dxx[j, i] = Dxx[j, i] - (dxi[1] + vBondPrime(dist) / dist)
            Dxx[i, j] = Dxx[i, j] + (dxj[1] - vBondPrime(dist) / dist)
            Dxy[j, i] = Dxy[j, i] - dxi[2]
            Dxy[i, j] = Dxy[i, j] + dxj[2]
            Dxz[j, i] = Dxz[j, i] - dxi[3]
            Dxz[i, j] = Dxz[i, j] + dxj[3]

            ###  With respect to y
            dyj = v .* (v[2] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dyi = v .* (v[2] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dyy[j, j] = Dyy[j, j] - dyj[2] + vBondPrime(dist) / dist
            Dyy[i, i] = Dyy[i, i] + dyi[2] + vBondPrime(dist) / dist
            Dyz[j, j] = Dyz[j, j] - dyj[3]
            Dyz[i, i] = Dyz[i, i] + dyi[3]

            ### Mixed derivatives
            Dyy[j, i] = Dyy[j, i] - (dyi[2] + vBondPrime(dist) / dist)
            Dyy[i, j] = Dyy[i, j] + (dyj[2] - vBondPrime(dist) / dist)
            Dyz[j, i] = Dyz[j, i] - dyi[3]
            Dyz[i, j] = Dyz[i, j] + dyj[3]

            ### with respect to z
            dzj = v .* (v[3] / (dist^2)) * (-vBondPrimePrime(dist) + vBondPrime(dist) / dist)
            dzi = v .* (v[3] / (dist^2)) * (vBondPrimePrime(dist) - vBondPrime(dist) / dist)

            Dzz[j, j] = Dzz[j, j] - dzj[3] + vBondPrime(dist) / dist
            Dzz[i, i] = Dzz[i, i] + dzi[3] + vBondPrime(dist) / dist

            ### Mixed derivatives
            Dzz[j, i] = Dzz[j, i] - (dzi[3] + vBondPrime(dist) / dist)
            Dzz[i, j] = Dzz[i, j] + (dzj[3] - vBondPrime(dist) / dist)

        end

    end


    DFx = interval(0.5) * [Dxx Dxy Dxz; Dxy Dyy Dyz; Dxz Dyz Dzz]

    ### Prelocating Matrices
    Dxx = zeros(type, N, N)
    Dxy = zeros(type, N, N)
    Dxz = zeros(type, N, N)

    Dyx = zeros(type, N, N)
    Dyy = zeros(type, N, N)
    Dyz = zeros(type, N, N)
    Dzx = zeros(type, N, N)
    Dzy = zeros(type, N, N)
    Dzz = zeros(type, N, N)


    ### ANGLE INTERACTION HESSIAN 
    vAnglePrime = a -> interval(2) * kθ * (a - θ)
    vAnglePrimePrime = a -> interval(2) * kθ


    for j = 1:N

        for k = 1:3

            ### Neighbour atoms 
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]

            ### These are vi and vk
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]

            ### These are vi and vk
            dv1_0 = interval.([-1, -1, -1])
            dv1_1 = interval.([1, 1, 1])
            dv1_2 = interval.([0, 0, 0])

            dv2_0 = interval.([-1, -1, -1])
            dv2_1 = interval.([0, 0, 0])
            dv2_2 = interval.([1, 1, 1])

            ### Norms squared
            v1v1 = dot(v1, v1)
            v2v2 = dot(v2, v2)

            ### Derivatives of v1v1 v2v2
            dv1v1_0 = -interval(2) * v1
            dv1v1_1 = interval(2) * v1
            dv1v1_2 = interval.([0, 0, 0])

            dv2v2_0 = -interval(2) * v2
            dv2v2_1 = interval.([0, 0, 0])
            dv2v2_2 = interval.(2) * v2

            ### Norm nrm
            nrm = sqrt(v1v1 * v2v2)

            ### Derivatives of nrm (dnrm_0 = dnrm1 + dnrm2)        
            dnrm_1 = (v2v2 / nrm) * v1
            dnrm_2 = (v1v1 / nrm) * v2
            dnrm_0 = -dnrm_1 - dnrm_2

            ### Angle and proj
            proj = dot(v1, v2) / nrm
            rad = sqrt(interval(1.0) - proj^2)
            rad2 = interval(1.0) - proj^2

            angle = acos(proj)

            ### Derivatives of proj(dproj_0 = -dv1-dv2 )
            dproj_1 = v2 / nrm - (proj / v1v1) * v1
            dproj_2 = v1 / nrm - (proj / v2v2) * v2
            dproj_0 = -dproj_1 .- dproj_2

            ### u1 and u2
            u1 = v1 / v1v1
            u2 = v2 / v2v2

            ### Derivatives of u1 and u2
            du1_0 = dv1_0 / v1v1 - (interval(1) / (v1v1^2)) * v1 .* dv1v1_0
            du1_1 = dv1_1 / v1v1 - (interval(1) / (v1v1^2)) * v1 .* dv1v1_1
            du1_2 = dv1_2 / v1v1 - (interval(1) / (v1v1^2)) * v1 .* dv1v1_2
            du2_0 = dv2_0 / v2v2 - (interval(1) / (v2v2^2)) * v2 .* dv2v2_0
            du2_1 = dv2_1 / v2v2 - (interval(1) / (v2v2^2)) * v2 .* dv2v2_1
            du2_2 = dv2_2 / v2v2 - (interval(1) / (v2v2^2)) * v2 .* dv2v2_2


            a = [1, 1, 2, 2, 3, 3]
            b = [2, 3, 3, 1, 1, 2]

            ### BUILDING DERIVATIVES FOR dv1 
            ## dv1 wrt Xi  
            dproj_11 = dv2_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v2 - proj .* du1_1 - u1 .* dproj_1
            Dv_11 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_1) / rad2 + dproj_11) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_1) / rad2

            Dxx[idx1, idx1] = Dxx[idx1, idx1] + Dv_11[1]
            Dyy[idx1, idx1] = Dyy[idx1, idx1] + Dv_11[2]
            Dzz[idx1, idx1] = Dzz[idx1, idx1] + Dv_11[3]

            dproj_11mixed = -dnrm_1[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_1[b] * (v1v1^-2)) - u1[a] .* dproj_1[b]
            Dv_11mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_1[b] / rad2) + dproj_11mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_1[b] / rad2)

            Dxy[idx1, idx1] = Dxy[idx1, idx1] + Dv_11mixed[1]
            Dxz[idx1, idx1] = Dxz[idx1, idx1] + Dv_11mixed[2]
            Dyz[idx1, idx1] = Dyz[idx1, idx1] + Dv_11mixed[3]
            Dyx[idx1, idx1] = Dyx[idx1, idx1] + Dv_11mixed[4]
            Dzx[idx1, idx1] = Dzx[idx1, idx1] + Dv_11mixed[5]
            Dzy[idx1, idx1] = Dzy[idx1, idx1] + Dv_11mixed[6]

            ### dv1 wrt Xk
            dproj_12 = dv2_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v2 - proj .* du1_2 - u1 .* dproj_2
            Dv_12 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_2) / rad2 + dproj_12) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_2) / rad2

            Dxx[idx1, idx2] = Dxx[idx1, idx2] + Dv_12[1]
            Dyy[idx1, idx2] = Dyy[idx1, idx2] + Dv_12[2]
            Dzz[idx1, idx2] = Dzz[idx1, idx2] + Dv_12[3]

            dproj_12mixed = -dnrm_2[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_2[b] * (v1v1^-2)) - u1[a] .* dproj_2[b]
            Dv_12mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_2[b] / rad2) + dproj_12mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_2[b] / rad2)

            Dxy[idx1, idx2] = Dxy[idx1, idx2] + Dv_12mixed[1]
            Dxz[idx1, idx2] = Dxz[idx1, idx2] + Dv_12mixed[2]
            Dyz[idx1, idx2] = Dyz[idx1, idx2] + Dv_12mixed[3]
            Dyx[idx1, idx2] = Dyx[idx1, idx2] + Dv_12mixed[4]
            Dzx[idx1, idx2] = Dzx[idx1, idx2] + Dv_12mixed[5]
            Dzy[idx1, idx2] = Dzy[idx1, idx2] + Dv_12mixed[6]

            ### dv1 wrt Xj   
            dproj_10 = dv2_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v2 - proj .* du1_0 - u1 .* dproj_0
            Dv_10 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_10) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[idx1, j] = Dxx[idx1, j] + Dv_10[1]
            Dyy[idx1, j] = Dyy[idx1, j] + Dv_10[2]
            Dzz[idx1, j] = Dzz[idx1, j] + Dv_10[3]

            dproj_10mixed = -dnrm_0[b] .* v2[a] * (nrm^-2) + proj * (v1[a] .* dv1v1_0[b] * (v1v1^-2)) - u1[a] .* dproj_0[b]
            Dv_10mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_1[a] .* dproj_0[b] / rad2) + dproj_10mixed) + vAnglePrimePrime(angle) * (dproj_1[a] .* dproj_0[b] / rad2)

            Dxy[idx1, j] = Dxy[idx1, j] + Dv_10mixed[1]
            Dxz[idx1, j] = Dxz[idx1, j] + Dv_10mixed[2]
            Dyz[idx1, j] = Dyz[idx1, j] + Dv_10mixed[3]
            Dyx[idx1, j] = Dyx[idx1, j] + Dv_10mixed[4]
            Dzx[idx1, j] = Dzx[idx1, j] + Dv_10mixed[5]
            Dzy[idx1, j] = Dzy[idx1, j] + Dv_10mixed[6]

            ### BUILDING DERIVATIVES FOR dv2 
            ### dv2 wrt Xi
            dproj_21 = dv1_1 ./ nrm - (nrm^-2) .* dnrm_1 .* v1 - proj .* du2_1 - u2 .* dproj_1
            Dv_21 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_1) / rad2 + dproj_21) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_1) / rad2

            Dxx[idx2, idx1] = Dxx[idx2, idx1] + Dv_21[1]
            Dyy[idx2, idx1] = Dyy[idx2, idx1] + Dv_21[2]
            Dzz[idx2, idx1] = Dzz[idx2, idx1] + Dv_21[3]

            dproj_21mixed = -dnrm_1[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_1[b] * (v2v2^-2)) - u2[a] .* dproj_1[b]
            Dv_21mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_1[b] / rad2) + dproj_21mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_1[b] / rad2)

            Dxy[idx2, idx1] = Dxy[idx2, idx1] + Dv_21mixed[1]
            Dxz[idx2, idx1] = Dxz[idx2, idx1] + Dv_21mixed[2]
            Dyz[idx2, idx1] = Dyz[idx2, idx1] + Dv_21mixed[3]
            Dyx[idx2, idx1] = Dyx[idx2, idx1] + Dv_21mixed[4]
            Dzx[idx2, idx1] = Dzx[idx2, idx1] + Dv_21mixed[5]
            Dzy[idx2, idx1] = Dzy[idx2, idx1] + Dv_21mixed[6]

            ### dv2 wrt Xk
            dproj_22 = dv1_2 ./ nrm - (nrm^-2) .* dnrm_2 .* v1 - proj .* du2_2 - u2 .* dproj_2
            Dv_22 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_2) / rad2 + dproj_22) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_2) / rad2

            Dxx[idx2, idx2] = Dxx[idx2, idx2] + Dv_22[1]
            Dyy[idx2, idx2] = Dyy[idx2, idx2] + Dv_22[2]
            Dzz[idx2, idx2] = Dzz[idx2, idx2] + Dv_22[3]

            dproj_22mixed = -dnrm_2[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_2[b] * (v2v2^-2)) - u2[a] .* dproj_2[b]
            Dv_22mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_2[b] / rad2) + dproj_22mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_2[b] / rad2)

            Dxy[idx2, idx2] = Dxy[idx2, idx2] + Dv_22mixed[1]
            Dxz[idx2, idx2] = Dxz[idx2, idx2] + Dv_22mixed[2]
            Dyz[idx2, idx2] = Dyz[idx2, idx2] + Dv_22mixed[3]
            Dyx[idx2, idx2] = Dyx[idx2, idx2] + Dv_22mixed[4]
            Dzx[idx2, idx2] = Dzx[idx2, idx2] + Dv_22mixed[5]
            Dzy[idx2, idx2] = Dzy[idx2, idx2] + Dv_22mixed[6]

            ### dv2 wrt Xj  
            dproj_20 = dv1_0 ./ nrm - (nrm^-2) .* dnrm_0 .* v1 - proj .* du2_0 - u2 .* dproj_0
            Dv_20 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_20) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2

            Dxx[idx2, j] = Dxx[idx2, j] + Dv_20[1]
            Dyy[idx2, j] = Dyy[idx2, j] + Dv_20[2]
            Dzz[idx2, j] = Dzz[idx2, j] + Dv_20[3]

            dproj_20mixed = -dnrm_0[b] .* v1[a] * (nrm^-2) + proj * (v2[a] .* dv2v2_0[b] * (v2v2^-2)) - u2[a] .* dproj_0[b]
            Dv_20mixed = -(vAnglePrime(angle) / rad) * (proj * (dproj_2[a] .* dproj_0[b] / rad2) + dproj_20mixed) + vAnglePrimePrime(angle) * (dproj_2[a] .* dproj_0[b] / rad2)

            Dxy[idx2, j] = Dxy[idx2, j] + Dv_20mixed[1]
            Dxz[idx2, j] = Dxz[idx2, j] + Dv_20mixed[2]
            Dyz[idx2, j] = Dyz[idx2, j] + Dv_20mixed[3]
            Dyx[idx2, j] = Dyx[idx2, j] + Dv_20mixed[4]
            Dzx[idx2, j] = Dzx[idx2, j] + Dv_20mixed[5]
            Dzy[idx2, j] = Dzy[idx2, j] + Dv_20mixed[6]


            ### BUILDING DERIVATIVES FOR dv0 
            ### dv0 wrt Xi

            dproj_01 = -dproj_11 - dproj_21
            Dv_01 = -(vAnglePrime(angle) / rad) * (proj * (dproj_1 .* dproj_0) / rad2 + dproj_01) + vAnglePrimePrime(angle) * (dproj_1 .* dproj_0) / rad2

            Dxx[j, idx1] = Dxx[j, idx1] + Dv_01[1]
            Dyy[j, idx1] = Dyy[j, idx1] + Dv_01[2]
            Dzz[j, idx1] = Dzz[j, idx1] + Dv_01[3]
            Dxy[j, idx1] = Dxy[j, idx1] - Dv_11mixed[1] - Dv_21mixed[1]
            Dxz[j, idx1] = Dxz[j, idx1] - Dv_11mixed[2] - Dv_21mixed[2]
            Dyz[j, idx1] = Dyz[j, idx1] - Dv_11mixed[3] - Dv_21mixed[3]
            Dyx[j, idx1] = Dyx[j, idx1] - Dv_11mixed[4] - Dv_21mixed[4]
            Dzx[j, idx1] = Dzx[j, idx1] - Dv_11mixed[5] - Dv_21mixed[5]
            Dzy[j, idx1] = Dzy[j, idx1] - Dv_11mixed[6] - Dv_21mixed[6]



            ### dv0 wrt Xk
            dproj_02 = -dproj_12 - dproj_22
            Dv_02 = -(vAnglePrime(angle) / rad) * (proj * (dproj_2 .* dproj_0) / rad2 + dproj_02) + vAnglePrimePrime(angle) * (dproj_2 .* dproj_0) / rad2


            Dxx[j, idx2] = Dxx[j, idx2] + Dv_02[1]
            Dyy[j, idx2] = Dyy[j, idx2] + Dv_02[2]
            Dzz[j, idx2] = Dzz[j, idx2] + Dv_02[3]
            Dxy[j, idx2] = Dxy[j, idx2] - Dv_12mixed[1] - Dv_22mixed[1]
            Dxz[j, idx2] = Dxz[j, idx2] - Dv_12mixed[2] - Dv_22mixed[2]
            Dyz[j, idx2] = Dyz[j, idx2] - Dv_12mixed[3] - Dv_22mixed[3]
            Dyx[j, idx2] = Dyx[j, idx2] - Dv_12mixed[4] - Dv_22mixed[4]
            Dzx[j, idx2] = Dzx[j, idx2] - Dv_12mixed[5] - Dv_22mixed[5]
            Dzy[j, idx2] = Dzy[j, idx2] - Dv_12mixed[6] - Dv_22mixed[6]

            ### dv0 wrt Xj      
            dproj_00 = -dproj_10 - dproj_20
            Dv_00 = -(vAnglePrime(angle) / rad) * (proj * (dproj_0 .* dproj_0) / rad2 + dproj_00) + vAnglePrimePrime(angle) * (dproj_0 .* dproj_0) / rad2

            Dxx[j, j] += Dv_00[1]
            Dyy[j, j] += Dv_00[2]
            Dzz[j, j] += Dv_00[3]
            Dxy[j, j] += -Dv_10mixed[1] - Dv_20mixed[1]
            Dxz[j, j] += -Dv_10mixed[2] - Dv_20mixed[2]
            Dyz[j, j] += -Dv_10mixed[3] - Dv_20mixed[3]
            Dyx[j, j] += -Dv_10mixed[4] - Dv_20mixed[4]
            Dzx[j, j] += -Dv_10mixed[5] - Dv_20mixed[5]
            Dzy[j, j] += -Dv_10mixed[6] - Dv_20mixed[6]

        end

    end

    dFx = DFx + [Dxx Dxy Dxz; Dyx Dyy Dyz; Dzx Dzy Dzz]

    ### Extending derivative
    dFx = [zeros(type, 3 * N, 6) dFx]
    dFx = [zeros(type, 6, 3 * N + 6); dFx]

    Adjusts = [zeros(type, N, N) -lambda1*Diagonal(ones(type, N)) lambda3*Diagonal(ones(type, N)); lambda1*Diagonal(ones(type, N)) zeros(type, N, N) -lambda2*Diagonal(ones(type, N)); -lambda3*Diagonal(ones(type, N)) lambda2*Diagonal(ones(type, N)) zeros(type, N, N)]
    Adjusts = [I1x I2x I3x T1 T2 T3 Adjusts]
    Adjusts = [zeros(type, 1, 6) transpose(I1xfix); zeros(type, 1, 6) transpose(I2xfix); zeros(type, 1, 6) transpose(I3xfix); zeros(type, 1, 6) transpose(T1); zeros(type, 1, 6) transpose(T2); zeros(type, 1, 6) transpose(T3); Adjusts]

    dFx = dFx + Adjusts

    return dFx

end

function Energy_Grad(x, connectivity, b, θ, kb, kθ)

    x = reshape(x, :, 3)
    n = size(connectivity, 1)
    e = zero(typeof(x[1]))
    de = zeros(typeof(x[1]), n, 3)

    ### Potential
    vBond = d -> kb * (d - b)^2
    vBondPrime = d -> 2 * kb * (d - b)

    vAngle = a -> kθ * (a - θ)^2
    vAnglePrime = a -> 2 * kθ * (a - θ)

    ### Pairwise interactions
    for j = 1:n
        for k = 1:3
            v = x[connectivity[j, k], :] - x[j, :]
            dist = sqrt(dot(v, v))
            e += vBond(dist)

            de[j, :] .+= (vBondPrime(dist) / dist) .* (-v)
            de[connectivity[j, k], :] .+= (vBondPrime(dist) / dist) .* v
        end
    end

    e = 0.5 * e
    de = 0.5 * de

    ### Angle interactions
    for j = 1:n
        for k = 1:3
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm
            angle = acos(proj)
            e += vAngle(angle)


            dv1 = v2 / nrm - (proj / dot(v1, v1)) * v1
            dv2 = v1 / nrm - (proj / dot(v2, v2)) * v2

            de[j, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) * (-dv1 - dv2)
            de[idx1, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv1
            de[idx2, :] += -(vAnglePrime(angle) / sqrt(1 - proj^2)) .* dv2

        end
    end

    de = de[:]

    return e, de


end

function Energy_Grad(x, connectivity, b::IntervalArithmetic.Interval, θ::IntervalArithmetic.Interval, kb::IntervalArithmetic.Interval, kθ::IntervalArithmetic.Interval)

    x = reshape(x, :, 3)
    n = size(connectivity, 1)
    e = zero(typeof(x[1]))
    de = zeros(typeof(x[1]), n, 3)

    ### Potential
    vBond = d -> kb * (d - b)^2
    vBondPrime = d -> interval(2) * kb * (d - b)

    vAngle = a -> kθ * (a - θ)^2
    vAnglePrime = a -> interval(2) * kθ * (a - θ)

    ### Pairwise interactions
    for j = 1:n
        for k = 1:3
            v = x[connectivity[j, k], :] - x[j, :]
            dist = sqrt(dot(v, v))
            e += vBond(dist)

            de[j, :] .+= (vBondPrime(dist) / dist) .* (-v)
            de[connectivity[j, k], :] .+= (vBondPrime(dist) / dist) .* v
        end
    end

    e = interval(0.5) * e
    de = interval(0.5) * de

    ### Angle interactions
    for j = 1:n
        for k = 1:3
            idx1 = connectivity[j, k]
            idx2 = connectivity[j, mod(k, 3)+1]
            v1 = x[idx1, :] - x[j, :]
            v2 = x[idx2, :] - x[j, :]
            nrm = sqrt(dot(v1, v1) * dot(v2, v2))
            proj = dot(v1, v2) / nrm
            angle = acos(proj)
            e += vAngle(angle)


            dv1 = v2 / nrm - (proj / dot(v1, v1)) * v1
            dv2 = v1 / nrm - (proj / dot(v2, v2)) * v2

            de[j, :] += -(vAnglePrime(angle) / sqrt(interval(1) - proj^2)) * (-dv1 - dv2)
            de[idx1, :] += -(vAnglePrime(angle) / sqrt(interval(1) - proj^2)) .* dv1
            de[idx2, :] += -(vAnglePrime(angle) / sqrt(interval(1) - proj^2)) .* dv2

        end
    end

    de = de[:]

    return e, de


end