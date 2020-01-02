#=
Readable, inefficient, rather dumb implementation of the simplex algorithm

min c'x
st  Ax = b
    x >= 0

At each iteration B denotes the basic columns of a, N the nonbasic columns,
i.e. A = [B | N]

(1) calculate the duals pi' = c_b' * B_inv
(2) calculate reduced costs c_n' - pi' * N
(3)
    (a) if all reduced costs are nonnegative, terminate
    (b) otherwise, find variable i = argmin_k{c_k - pi' * A_k} with the smallest reduced cost
(4) compute B_inv * A_i
(5) find the leaving variable
    j = argmin_k{e_k' * B_inv * b / e_k' * B_inv * A_i : e_k' * B_inv * A_i > 0}
    (a) if e_k' * B_inv * A_i <= 0 for all k, return extreme ray using B_inv * A_i
    (b) otherwise, variable x_j leaves the basis
(6) update the basis
    remember B is never needed in the algorithm, only B_inv
    let Q be the elementary matrix such that Q * B_inv * A_i = e_j
    then update B_inv * b and B_inv by applying Q


* views
* don't even need a copy of x_b, make it a view, maybe dumb version can have an array we copy into
actually nah too ugly. for both x and c_b we can maintain views. make it the first
improvement that we will do. will need a preassignment to quiz on steps of
simplex or something to refresh the memory.

assumes b >= 0 for phase I to work
assumes A doesn't have redundant constraints while problem is feasible (won't be eliminating constraints after Phase I)

=#
using LinearAlgebra # for dot
using TimerOutputs
const TO = TimerOutput()

function find_entering_var(A::Matrix{T}, c::Vector{T}, pi::Vector{T}, var_status::Vector{Int}, phase_I_data::Bool) where {T <: Real}
    min_rc = 0
    min_idx = 0
    for k in eachindex(var_status)
        # only check nonbasic variables
        if iszero(var_status[k])
            ck = (phase_I_data ? 0 : c[k])
            @timeit TO "rc" @views rc = ck - sum(A[i, k] * pi[i] for i in eachindex(pi)) # dot(A[:, k], pi)
            if rc < min_rc
                min_rc = rc
                min_idx = k
            end
        end
    end
    return (min_rc, min_idx)
end

function find_leaving_var(n::Int, x_b::Vector{T}, B_inv_A_i::Vector{T}, basic_idxs::Vector{Int}, phase_I_data::Bool) where {T <: Real}
    min_ratio = T(Inf)
    min_idx = 0
    for k in eachindex(B_inv_A_i)
        if B_inv_A_i[k] > 0
            # drive the first artificial variable out of the basis if any remain
            if !phase_I_data && basic_idxs[k] > n
                return (zero(T), k)
            end
            ratio = x_b[k] / B_inv_A_i[k]
            if ratio < min_ratio
                min_ratio = ratio
                min_idx = k
            end
        end
    end
    return (min_ratio, min_idx)
end

function basic_data_update(var_status::Vector{Int}, basic_idxs::Vector{Int}, c_b::Vector{Float64}, c::Vector{Float64}, entering_ind::Int, leaving_ind::Int, phase_I_data::Bool)
    basic_idxs[leaving_ind] = entering_ind
    var_status[var_status .== leaving_ind] .= 0
    var_status[entering_ind] = leaving_ind
    c_b[leaving_ind] = (phase_I_data ? 0 : c[entering_ind])
    return
end

function tableau_update(B_inv::Matrix{Float64}, B_inv_A_i::Vector{Float64}, x_b::Vector{Float64}, leaving_ind::Int)
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

function fullrsm(A::Matrix{Float64}, b::Vector{Float64}, c::Vector{Float64})
    (m, n) = size(A)
    entering_ind = 0
    @assert all(bi -> bi >= 0, b)

    # set basic variables to artificial variables
    @timeit TO "setup" begin
    basic_idxs = collect((n + 1):(m + n))
    var_status = zeros(Int, n)
    B_inv = Matrix{Float64}(I, m, m)
    x_b = copy(b)
    c_b = ones(m)
    phase_I_data = true
    x = zeros(n)
    pi = zeros(m)
    obj = NaN
    unbounded = false
    finished_rsm = false
    B_inv_A_i = zeros(m)
    end

    @timeit TO "rsm" while !finished_rsm
        while true
            # update duals
            @timeit TO "pi" pi = B_inv' * c_b

            # find an entering variable
            @timeit TO "entering" (min_rc, entering_ind) = find_entering_var(A, c, pi, var_status, phase_I_data)
            # if all reduced costs are nonnegative, we found an optimal solution to the current phase
            if iszero(entering_ind)
                break
            end

            # find a leaving variable
            @timeit TO "BiAi" B_inv_A_i = B_inv * A[:, entering_ind]
            (min_ratio, leaving_ind) = find_leaving_var(n, x_b, B_inv_A_i, basic_idxs, phase_I_data)
            if iszero(leaving_ind)
                @assert !phase_I_data
                unbounded = true
                break
            end

            # update data
            @timeit TO "tableau" tableau_update(B_inv, B_inv_A_i, x_b, leaving_ind)
            basic_data_update(var_status, basic_idxs, c_b, c, entering_ind, leaving_ind, phase_I_data)
        end

        # compute objective
        phase_obj = dot(c_b, x_b)

        if phase_I_data && phase_obj > 0
            # problem is not feasible
            obj = Inf
            pi .= 0
            finished_rsm = true
        elseif phase_I_data
            # problem is feasible, and ready to move to phase II
            phase_I_data = false
            # update c_b with actual costs of the original problem variables that are in the basis
            for (k, idx) in enumerate(basic_idxs)
                if idx <= n
                    c_b[k] = c[idx]
                end
            end
        # stop if we have arrived at the end of pahse II
        else
            finished_rsm = true
            if unbounded
                # return an extreme ray
                x[basic_idxs] .= -B_inv_A_i
                x[entering_ind] = 1
                obj = -Inf
            else
                x[basic_idxs] .= x_b
                obj = phase_obj
            end
        end
    end
    return (x, pi, obj)
end
