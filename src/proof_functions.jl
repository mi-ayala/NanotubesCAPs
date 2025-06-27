### Validation functions

function get_proof(x₀, F, DF, r0)

    DF_interval = mid.(DF(interval.(x₀)))
    try
        A = inv(DF_interval)
    catch err
        @warn "Could not invert mid.(DF(interval.(x₀)))."
        return nothing
    end

    A = inv(DF_interval)

    x₀_interval = interval.(x₀)
    F_interval = F(x₀_interval)
    Y = norm(interval.(A) * F_interval, Inf)


    x₀r_interval = interval.(x₀, r0; format=:midpoint)
    DF_interval = DF(x₀r_interval)
    Z₁ = opnorm(LinearOperator(interval.(A) * DF_interval - UniformScaling(interval(1))), Inf)

    return interval_of_existence(Y, Z₁, r0)


end

function find_maximum_r_star(x₀, F, DF, r0;
    tol::Float64=1e-11,
    max_iters::Int=40
)

    consecutive = 1

    # Compute the midpoint derivative interval and its inverse
    DF_interval = mid.(DF(interval.(x₀)))
    try
        A = inv(DF_interval)
    catch err
        @warn "Could not invert mid.(DF(interval.(x₀)))."
        return nothing
    end

    A = inv(DF_interval)

    # Evaluate F and set up Y 
    x₀_interval = interval.(x₀)
    F_interval = F(x₀_interval)
    Y = norm(interval.(A) * F_interval, Inf)

    # Initialize the search bracket:
    r_left = sup(Y)
    r_right = r0

    # For stagnation detection, store previous r_left
    r_left_old = r_left
    stagnation_count = 0


    for i in 1:max_iters

        # Midpoint of the current bracket
        r_mid = 0.5 * (r_left + r_right)

        # Evaluate the function over an interval around x₀ with radius r_mid
        x₀r_interval = interval.(x₀, r_mid; format=:midpoint)
        DF_interval = DF(x₀r_interval)
        Z₁ = opnorm(LinearOperator(interval.(A) * DF_interval - UniformScaling(interval(1))), Inf)
        pass_proof = interval_of_existence(Y, Z₁, r_mid)

        # If proof passes at r_mid, update the lower bound r_left
        if inf(pass_proof) ≠ Inf
            r_left = r_mid

            if i > 1
                if abs(r_left_old - r_left) < 1e-11
                    stagnation_count += 1
                    println("Difference in r_star:")
                    println(abs(r_left - r_left_old))
                    if stagnation_count >= consecutive
                        @info "Stagnation detected: r* has plateaued after $i iterations."
                        break
                    end
                else
                    stagnation_count = 0
                end
            end



        else
            r_right = r_mid
        end

        # Binary search tolerance
        if (r_right - r_left) < tol
            @info "Bisection converged after $i iterations (bracket tol reached)."
            break
        end

        r_left_old = r_left
    end

    return (r_left, r_right)
end

function find_maximum_r_star(x₀, F, DF;
    tol::Float64=1e-11,
    max_iters::Int=40
)
    # Initial guess for r
    r0 = 1e-7
    consecutive = 1

    # Compute the midpoint derivative interval and its inverse
    DF_interval = mid.(DF(interval.(x₀)))
    try
        A = inv(DF_interval)
    catch err
        @warn "Could not invert mid.(DF(interval.(x₀)))."
        return nothing
    end

    A = inv(DF_interval)

    # Evaluate F and set up Y 
    x₀_interval = interval.(x₀)
    F_interval = F(x₀_interval)
    Y = norm(interval.(A) * F_interval, Inf)

    # Initialize the search bracket
    r_left = sup(Y)
    r_right = r0

    # For stagnation detection
    r_left_old = r_left
    stagnation_count = 0


    for i in 1:max_iters

        # Midpoint of the current bracket
        r_mid = 0.5 * (r_left + r_right)

        # Evaluate the function over an interval around x₀ with radius r_mid
        x₀r_interval = interval.(x₀, r_mid; format=:midpoint)
        DF_interval = DF(x₀r_interval)
        Z₁ = opnorm(LinearOperator(interval.(A) * DF_interval - UniformScaling(interval(1))), Inf)
        pass_proof = interval_of_existence(Y, Z₁, r_mid)

        # If proof passes at r_mid, update the lower bound r_left
        if inf(pass_proof) ≠ Inf
            r_left = r_mid

            if i > 1

                if abs(r_left - r_left_old) < 1e-11
                    stagnation_count += 1
                    println("Difference in r_star:")
                    println(abs(r_left - r_left_old))
                    if stagnation_count >= consecutive
                        @info "Stagnation detected: r* has plateaued after $i iterations."
                        break
                    end
                else
                    stagnation_count = 0
                end
            end

        else
            r_right = r_mid
        end

        # Check the standard binary search tolerance
        if (r_right - r_left) < tol
            @info "Bisection converged after $i iterations (bracket tol reached)."
            break
        end


        r_left_old = r_left
    end

    return (r_left, r_right)
end

function find_maximum_r_star(x₀, F, DF, r_works, r_fails;
    tol::Float64=1e-11,
    max_iters::Int=40
)

    consecutive = 2

    # Compute the midpoint derivative interval and its inverse
    DF_interval = mid.(DF(interval.(x₀)))
    try
        A = inv(DF_interval)
    catch err
        @warn "Could not invert mid.(DF(interval.(x₀)))."
        return nothing
    end

    A = inv(DF_interval)

    # Evaluate F and set up Y 
    x₀_interval = interval.(x₀)
    F_interval = F(x₀_interval)
    Y = norm(interval.(A) * F_interval, Inf)

    # Initialize the search bracket:
    r_left = r_works
    r_right = r_fails

    # For stagnation detection, store previous r_left
    r_left_old = r_left
    stagnation_count = 0


    for i in 1:max_iters

        # Midpoint of the current bracket
        r_mid = 0.5 * (r_left + r_right)

        # Evaluate the function over an interval around x₀ with radius r_mid
        x₀r_interval = interval.(x₀, r_mid; format=:midpoint)
        DF_interval = DF(x₀r_interval)
        Z₁ = opnorm(LinearOperator(interval.(A) * DF_interval - UniformScaling(interval(1))), Inf)
        pass_proof = interval_of_existence(Y, Z₁, r_mid)

        # If proof passes at r_mid, update the lower bound r_left
        if inf(pass_proof) ≠ Inf
            r_left = r_mid

            println("Difference in r_star:")
            println(abs(r_left - r_left_old))

            if i > 1
                if abs(r_left - r_left_old) < 1e-11
                    stagnation_count += 1
                    println("Difference in r_star:")
                    println(abs(r_left - r_left_old))
                    if stagnation_count >= consecutive
                        @info "Stagnation detected: r* has plateaued after $i iterations."
                        break
                    end
                else
                    stagnation_count = 0
                end
            end



        else
            r_right = r_mid
        end

        # Binary search tolerance
        if (r_right - r_left) < tol
            @info "Bisection converged after $i iterations (bracket tol reached)."
            break
        end

        r_left_old = r_left
    end

    return (r_left, r_right)
end

function newton_method(F_DF::Function, x₀; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=true)
    # Validate input parameters
    if tol <= 0 || maxiter <= 0
        throw(DomainError((tol, maxiter), "Tolerance and maximum number of iterations must be positive"))
    end

    if verbose
        println("Newton's method: Inf-norm, tol = $tol, maxiter = $maxiter")
        println("      iteration        |F(x)|")
        println("-------------------------------------")
    end

    x = copy(x₀)
    F, DF = F_DF(x)
    nF = norm(F, Inf)
    if verbose
        @printf("%11d %19.4e\n", 0, nF)
    end
    if nF <= tol
        return x, true, 0
    end

    # Newton's iterations using a for-loop
    for i in 1:maxiter
        # Compute Newton update (delta): DF * delta = F
        delta = DF \ F
        if verbose
            abs_delta = norm(delta, Inf)
            rel_delta = abs_delta / max(norm(x, Inf), eps())
             @printf(" |delta| = %10.4e, relative delta = %10.4e\n", abs_delta, rel_delta)
        end
        x .-= delta

        F, DF = F_DF(x)
        nF = norm(F, Inf)
        if verbose
             @printf("%11d %19.4e\n", i, nF)
        end

        if nF <= tol
            return x, true, i
        end
    end

    return x, false, maxiter
end


function common_decimal_places(nums::AbstractVector{<:AbstractFloat})::Int

    if length(nums) ≤ 1
        return 15  # or 0, depending on your preference
    end

    mn = minimum(nums)
    mx = maximum(nums)

    if mx == mn
        return 15
    end


    diff = mx - mn

    d = floor(Int, -log10(diff))

    d = max(d, 0)

    d = min(d, 15)

    return d
end