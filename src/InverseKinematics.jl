module InverseKinematics

using LinearAlgebra,
      Statistics

using LinearAlgebra: norm_sqr

using Rotations, CoordinateTransformations, StaticArrays

export kabsch!,
       fa3r!

struct SegOpt{T}; end
struct Kabsch; end
struct FA3R{T}
    maxiter::Int
    ϵ::T
end

SegOpt(::T) where T = SegOpt{T}()
SegOpt() = SegOpt{Kabsch}()
FA3R() = FA3R{Float64}(100, 0.0)

struct IKSeg{S,T,L}
    _def::SVector{L,T}
    origin::SVector{3,T}
    ori::Quat{T}
    prox::SVector{3,T}
    dist::SVector{3,T}
end

function IKSeg(def::SVector{L,T}, ori::Quat{T}, prox::SVector{3,T}, dist::SVector{3,T}) where {L, T}
    if !iszero(L % 3)
        throw(ArgumentError("def must be a multiple of 3"))
    else
        origin = mean(reinterpret(SVector{3,T}, def))
        return IKSeg{fld(L,3),T,L}(def, origin, ori, prox, dist)
    end
end

function getproperty(s::IKSeg{N,T}, b) where {N,T}
    if b == :def
        return reinterpret(SVector{3,T},s._def)
    else
        return getfield(s,b)
    end
end

struct IKModel{T}
end

struct IKSegResult{T}
    ori::Vector{Quat{T}}
    pos::Vector{SVector{3,T}}
    e::Vector{T}
end

function IKResult(model::IKModel, data)
    rootseg = root(model)
    l = size(data[rootseg], 1)
    T = eltype(data[rootseg])
    ikr = Dict{Symbol,IKSegResult{T}}()

    for seg in model
        ikr[seg] = IKSegResult{T}(Vector{Quat{T}}(undef, l),
                                  Vector{SVector{3,T}}(undef, l),
                                  Vector{T}(undef, l))
    end

    return ikr
end

"""
    inversekinematics(alg, model, data)

Calculate the kinematics of `model` using `data` and the specified algorithm, `alg`.
"""
function inversekinematics(::SegOpt{Kabsch}, model, data)
    ikdata = IKResult(model, data)
    for seg in model
        segikori = view(ikdata[seg].ori)
        segikpos = view(ikdata[seg].pos)
        segike = view(ikdata[seg].e)

        tmp1 = similar(seg.def, SArray{Tuple{1,3},T,2,3})
        tmp2 = similar(seg.def, SArray{Tuple{3,1},T,2,3})
        Ht = similar(seg.def, SArray{Tuple{3,3},T,2,9})

        for (i, frame) in enumerate(data[seg])
            q, t, e = kabsch!(frame, seg.def, tmp1, tmp2, Ht)
            segikori[i] = q
            segikpos[i] = t
            segike[i] = e
        end
    end
    return ikdata
end

# Kabsch algorithm with minimal allocations, using buffers q and p
function kabsch!(Q, P, q, p, Ht)
    # Calculate point set's centroid
    q_cent, p_cent = mean(Q), mean(P)

    # Center each set of points
    map!(x -> x - p_cent, p, P)
    map!(x -> x - q_cent, q, Q)

    # Cross-covariance matrix
    broadcast!(*, Ht, p, q)
    H = sum(Ht)

    # SVD
    Hsvd = svd(H)
    V = Hsvd.V
    U = Hsvd.U

    # Correct for handedness (corrects to a right-handed system)
    Hp = V*U'
    di = SMatrix{3,3}(Diagonal(SVector{3}(1, 1, sign(det(Hp)))))

    R = V * di * U'
    T = q_cent - p_cent
    e = metricerror(Q, P, R)

    qt = Quat(R)

    return (qt, T, e)
end

function fa3r!(Q, P, q, p, Ht, maxk::Int=100, ϵ=0)
    # Calculate point set's centroid
    q_cent, p_cent = mean(Q), mean(P)

    # Center each set of points
    map!(x -> x - p_cent, p, P)
    map!(x -> x - q_cent, q, Q)

    # Cross-covariance matrix
    broadcast!(*, Ht, p, q)
    H = sum(Ht)

    # Iterative method a la. Wu et al. 2018
    vkx = H[:,1]
    vky = H[:,2]
    vkz = H[:,3]

    k = 0
    while k < maxk
        # ρₖ₋₁ = 2/(norm(vkx)^2 + norm(vky)^2 + norm(vkz)^2 + 1)
        ρₖ₋₁ = 2/(norm_sqr(vkx) + norm_sqr(vky) + norm_sqr(vkz) + 1)

        v1x = vkx
        v1y = vky
        v1z = vkz
        vkx = ρₖ₋₁ * (v1x + v1y × v1z)
        vky = ρₖ₋₁ * (v1y + v1z × v1x)
        vkz = ρₖ₋₁ * (v1z + v1x × v1y)

        # if (norm(vkx - v1x)^2 + norm(vky - v1y)^2 + norm(vkz - v1z)^2) < ϵ
        if (norm_sqr(vkx - v1x) + norm_sqr(vky - v1y) + norm_sqr(vkz - v1z)) < ϵ
            break
        end
        k += 1
    end

    R = SMatrix{3,3}(vkx[1], vky[1], vkz[1], vkx[2], vky[2], vkz[2], vkx[3], vky[3], vkz[3])
    T = q_cent - p_cent
    e = metricerror(q, p, R)

    qt = Quat(R)

    return (qt, T, e)
end

# Kabsch algorithm
"""
    fitseg(Q,P)

Find the optimal rotation between two sets of points `P` and `Q`, where `P` is the original orientation, and `Q` is the new orientation.
"""
function fitseg(Q, P)
    # Center both sets of points
    q = Q .- Ref(mean(Q))
    p = P .- Ref(mean(P))

    # Cross-covariance matrix
    H = sum(p .* transpose.(q))
    Hsvd = svd(H)

    # Correct for handedness (corrects to a right-handed system)
    Hp = Hsvd.V*Hsvd.U'
    di = SMatrix{3,3}(Diagonal(SVector{3}(1, 1, sign(det(Hp)))))

    R = Hsvd.V * di * Hsvd.U'
end

function metricerror(Q, P, R)
    e = zero(eltype(R))
    for i in eachindex(Q,P)
        e += norm_sqr(Q[i]' - R*P[i])
    end
    return e
end

function rmat(p1, p2, p3)
    v1 = unit(p1 - p2)
    v2 = unit(v1 × (p3 - p2))
    v3 = unit(v2 × v1)

    return [v1 v2 v3]
end

function unit(v)
    return v ./ norm(v)
end

end # module
