#=
Readable, inefficient, rather dumb implementation of the simplex algorithm

min c'x
st  Ax = b
    x >= 0

no phase I
=#
using LinearAlgebra # for dot
using TimerOutputs
const TO = TimerOutput()

function find_entering_var_2(A::Matrix{T}, c::Vector{T}, pi::Vector{T}, var_status::Vector{Int}) where {T <: Real}
    min_rc = zero(T)
    min_idx = 0
    for k in eachindex(var_status)
        # only check nonbasic variables
        if iszero(var_status[k])
            @timeit TO "rc" @views rc = c[k] - dot(A[:, k], pi)
            if rc < min_rc
                min_rc = rc
                min_idx = k
            end
        end
    end
    return (min_rc, min_idx)
end

function find_leaving_var_2(x_b::Vector{T}, B_inv_A_i::Vector{T}, basic_idxs::Vector{Int}) where {T <: Real}
    min_ratio = T(Inf)
    min_idx = 0
    for k in eachindex(B_inv_A_i)
        if B_inv_A_i[k] > 0
            ratio = x_b[k] / B_inv_A_i[k]
            if ratio < min_ratio
                min_ratio = ratio
                min_idx = k
            end
        end
    end
    return (min_ratio, min_idx)
end

function basic_data_update_2(var_status::Vector{Int}, basic_idxs::Vector{Int}, c_b::AbstractVector{T}, c::Vector{T}, entering_ind::Int, leaving_ind::Int) where {T <: Real}
    basic_idxs[leaving_ind] = entering_ind
    var_status[var_status .== leaving_ind] .= 0
    var_status[entering_ind] = leaving_ind
    # c_b[leaving_ind] = c[entering_ind] # needed if not using views
    return
end

function tableau_update_2(B_inv::Matrix{T}, B_inv_A_i::Vector{T}, x_b::Vector{T}, leaving_ind::Int) where {T <: Real}
    B_inv[leaving_ind, :] ./= B_inv_A_i[leaving_ind]
    x_b[leaving_ind] /= B_inv_A_i[leaving_ind]
    for k in eachindex(B_inv_A_i)
        if k != leaving_ind
            B_inv[k , :] .-= B_inv[leaving_ind, :] * B_inv_A_i[k]
            x_b[k] -= x_b[leaving_ind] * B_inv_A_i[k]
        end
    end
    return
end

function fullrsm_2(
    basic_idxs::Vector{Int},
    var_status::Vector{Int},
    A::Matrix{T},
    B_inv::Matrix{T},
    x_b::Vector{T},
    c::Vector{T},
    ) where {T}

    entering_ind = 0
    @views c_b = c[basic_idxs]
    B_inv_A_i = similar(x_b)
    obj = NaN
    unbounded = false
    x = similar(c)
    x .= 0
    pi = similar(x_b)

    while true
        # update duals
        @timeit TO "pi" pi = B_inv' * c_b

        # find an entering variable
        @timeit TO "entering" (min_rc, entering_ind) = find_entering_var_2(A, c, pi, var_status)
        # if all reduced costs are nonnegative, we found an optimal solution to the current phase
        if iszero(entering_ind)
            break
        end

        # find a leaving variable
        @timeit TO "BiAi" B_inv_A_i = B_inv * A[:, entering_ind]
        (min_ratio, leaving_ind) = find_leaving_var_2(x_b, B_inv_A_i, basic_idxs)
        if iszero(leaving_ind)
            unbounded = true
            break
        end

        # update data
        @timeit TO "tableau" tableau_update_2(B_inv, B_inv_A_i, x_b, leaving_ind)
        basic_data_update_2(var_status, basic_idxs, c_b, c, entering_ind, leaving_ind)
    end

    if unbounded
        # return an extreme ray
        x[basic_idxs] .= -B_inv_A_i
        x[entering_ind] = 1
        obj = -Inf
    else
        x[basic_idxs] .= x_b
        obj = dot(c_b, x_b)
    end

    return (x, pi, obj)
end
