module InverseKinematics

using LinearAlgebra, Statistics
using LinearAlgebra: norm_sqr

using Rotations, CoordinateTransformations, StaticArrays

export kabsch, fa3r, fitseg, metricerror, rmat, unit

export SegOpt, Kabsch, FA3R, IKSeg, AbstractPoints, AbstractMaybePoints

struct SegOpt{T}
    alg::T
end
SegOpt() = SegOpt{Kabsch}(Kabsch())

struct Kabsch; end

struct FA3R{T}
    maxiters::Int
    ϵ::T
end
FA3R() = FA3R{Float32}(750, 1f-15)

AbstractPoints{T} = AbstractVector{SVector{3,T}}
AbstractMaybePoints{T} = AbstractVector{SVector{3,S}} where S <: Union{Missing, T}

struct IKSeg{N,T}
    def::SVector{N,SVector{3,T}}
    ori::UnitQuaternion{T}
    prox::SVector{3,T}
    dist::SVector{3,T}
end

struct IKModel{T}
end

struct IKSegResult{T} <: AbstractArray{T,1}
    ori::Vector{UnitQuaternion{T}}
    pos::Vector{SVector{3,T}}
    e::Vector{T}
end

function IKSegResult{T}(::UndefInitializer, len::Integer) where T
    return IKSegResult{T}(Vector{UnitQuaternion{T}}(undef, len),
                          Vector{SVector{3,T}}(undef, len),
                          Vector{T}(undef, len))
end

Base.size(x::IKSegResult) = size(x.e)
Base.length(x::IKSegResult) = length(x.e)

Base.getindex(x::IKSegResult, i::Int) = (;ori=x.ori[i], pos=x.pos[i], e=x.e[i])
Base.getindex(x::IKSegResult, i) = IKSegResult(x.ori[i], x.pos[i], x.e[i])

function Base.setindex!(
    x::IKSegResult{T},
    val::Tuple{UnitQuaternion{T},SVector{3,T},T},
    i
) where T
    x.ori[i] = val[1]
    x.pos[i] = val[2]
    x.e[i] = val[3]

    return x
end

function IKResult(model::IKModel, data)
    rootseg = root(model)
    l = size(data[rootseg], 1)
    T = eltype(data[rootseg])
    ikr = Dict{Symbol,IKSegResult{T}}()

    for seg in model
        ikr[seg] = IKSegResult{T}(Vector{UnitQuaternion{T}}(undef, l),
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
    for seg in model, (i, frame) in enumerate(data[seg])
        ikdata[seg][i] = kabsch(frame, seg.def)
    end
    return ikdata
end

function fitseg(
    ::Kabsch, Q, P::AbstractPoints{T}; w::AbstractVector{T}=fill!(similar(P, T), one(T))
) where T
    kabsch(Q, P, w)
end

function fitseg(
    f::FA3R, Q, P::AbstractPoints{T}; w::AbstractVector{T}=fill!(similar(P, T), one(T))
) where T
    fa3r(Q, P, w, f.maxiters, f.ϵ)
end

_throw_mismatch_pointset_lengths(qlen, plen) = throw(ArgumentError("`Q` and `P` must have" *
    "the same number of points; got length(Q) == $qlen and length(P) == $plen"))
_throw_mismatch_weight_length(len, wlen) = throw(ArgumentError("Weight vector must be the" *
    "same length as `Q` and `P`; got length(Q) == length(P) == $len and length(w) == $wlen"))

function kabsch(
    Q::AbstractMaybePoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T}=fill!(similar(P, T), one(T))
) where T
    length(Q) === length(P) || _throw_mismatch_pointset_lengths(length(Q), length(P))
    length(Q) === length(w) || _throw_mismatch_weight_length(length(Q), length(w))

    present = findpresent(Q)

    kabsch(Q, P, w, present)
end

function kabsch(
    Q::AbstractPoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T}=fill!(similar(P, T), one(T))
) where T
    length(Q) === length(P) || _throw_mismatch_pointset_lengths(length(Q), length(P))
    length(Q) === length(w) || _throw_mismatch_weight_length(length(Q), length(w))

    present = axes(Q, 1)

    kabsch(Q, P, w, present)
end

function kabsch(
    Q, P::AbstractPoints{T}, w::AbstractVector{T}, present::AbstractVector{Int}
)::Tuple{UnitQuaternion{T},SVector{3,T},T} where T
    S = length(present)
    if S < 3 # A full fit isn't possible
        return incompletefit(S, Q, P, w, present)
    end

    #Calculate point set's centroid
    q̄ = weightedmean(Q, w, present)
    p̄ = weightedmean(P, w, present)
    H = crosscov(Q, q̄, P, p̄, w, present)

    # SVD
    # Hsvd = svd(H)
    Hsvd = svd(Matrix(H); full=true, alg=LinearAlgebra.QRIteration())
    V = SMatrix{3,3}(Hsvd.V)
    U = SMatrix{3,3}(Hsvd.U)

    # Correct for handedness (corrects to a right-handed system)
    Hp = V*transpose(U)
    di = SMatrix{3,3}(Diagonal(SVector{3}(1, 1, sign(det(Hp)))))

    R = V * di * transpose(U)
    qt = UnitQuaternion(R)
    t = q̄ - qt*p̄
    e = metricerror(Q, P, w, t, present, qt)

    return (qt, t, e)
end

function fa3r(
    Q::AbstractMaybePoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T}=fill!(similar(P, T), one(T)),
    maxiters::Int=100, ϵ::T=0
) where T
    length(Q) === length(P) || _throw_mismatch_pointset_lengths()
    length(Q) === length(w) || _throw_mismatch_weight_length()

    present = findpresent(Q)

    fa3r(Q, P, w, present, maxiters, ϵ)
end

function fa3r(
    Q::AbstractPoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T}=fill!(similar(P, T), one(T)),
    maxiters::Int=100,
    ϵ::T=0
) where T
    length(Q) === length(P) || _throw_mismatch_pointset_lengths()
    length(Q) === length(w) || _throw_mismatch_weight_length()

    present = axes(Q, 1)

    fa3r(Q, P, w, present, maxiters, ϵ)
end

function fa3r(
    Q::AbstractMaybePoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T},
    present::AbstractVector{Int},
    maxiters::Int,
    ϵ::T
)::Tuple{UnitQuaternion{T},SVector{3,T},T} where T
    S = length(present)
    if S < 3 # A full fit isn't possible
        return incompletefit(S, Q, P, w, present)
    end

    #Calculate point set's centroid
    q̄ = weightedmean(Q, w, present)
    p̄ = weightedmean(P, w, present)
    H = crosscov(Q, q̄, P, p̄, w, present)

    # Iterative method a la. Wu et al. 2018
    vkx = H[:,1]
    vky = H[:,2]
    vkz = H[:,3]

    k = 0
    while k < maxiters
        ρₖ₋₁ = 2/(norm_sqr(vkx) + norm_sqr(vky) + norm_sqr(vkz) + 1)

        v1x = vkx
        v1y = vky
        v1z = vkz
        vkx = ρₖ₋₁ * (v1x + v1y × v1z)
        vky = ρₖ₋₁ * (v1y + v1z × v1x)
        vkz = ρₖ₋₁ * (v1z + v1x × v1y)

        if (norm_sqr(vkx - v1x) + norm_sqr(vky - v1y) + norm_sqr(vkz - v1z)) < ϵ
            break
        end
        k += 1
    end

    R = SMatrix{3,3}(vkx[1], vky[1], vkz[1], vkx[2], vky[2], vkz[2], vkx[3], vky[3], vkz[3])
    qt = UnitQuaternion(R)
    # qt = UnitQuaternion( vkx[1] + vky[2] + vkz[3],
    #           -vky[3] + vkz[2],
    #           -vkz[1] + vkx[3],
    #           -vkx[2] + vky[1])
    # qt = UnitQuaternion( vkx[1] + vky[2] + vkz[3],
    #           -vkz[2] + vky[3],
    #           -vkx[3] + vkz[1],
    #           -vky[1] + vkx[2])
    t = q̄ - qt*p̄
    e = metricerror(Q, P, w, t, present, qt)


    return (qt, t, e)
end

function findpresent(Q::AbstractMaybePoints{T}) where T
    misspt = SVector{3,Union{Missing,T}}(missing, missing, missing)
    boolpres = Q .!== Ref(misspt)
    N = sum(boolpres)
    present = MArray{Tuple{N},Int,1,N}(undef)

    j = 1
    for i in eachindex(Q)
        if boolpres[i]
            present[j] = i
            j += 1
        end
    end

    return present
end

function incompletefit(
    S::Int,
    Q::AbstractMaybePoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T},
    present::AbstractVector{Int}
) where T
    t = zero(SVector{3, T})

    if S > 0 # We can get a very rough position
        q_cent = weightedmean(Q, w, present)
        p_cent = weightedmean(P, w, present)

        t = q_cent - p_cent
    end

    qt = one(UnitQuaternion{T})
    e = T(NaN)

    return (qt, t, e)
end

function weightedmean(X, w::AbstractVector{T}, present::AbstractVector{Int}) where T
    center = zero(SVector{3,T})
    for i in present
        center = SVector{3,T}(X[i])*w[i] + center
    end

    return center/sum(w)
end

function crosscov(
    Q,
    q̄::SVector{3,T},
    P::AbstractPoints{T},
    p̄::SVector{3,T},
    w::AbstractVector{T},
    present::AbstractVector{Int}
) where T
    ŵ = sum(w)

    H = zero(SArray{Tuple{3,3},T})
    for i in present
        p′ = SArray{Tuple{3,1},T}(P[i] - p̄)
        q′ = SArray{Tuple{1,3},T}(SVector{3,T}(Q[i]) - q̄)

        # Cross-covariance matrix
        H = H + (p′ * q′)*w[i]
    end
    H = H#/ŵ

    return H
end

function metricerror(
    Q::AbstractPoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T},
    t::SVector{3,T},
    R::Rotation{3,T}
) where T
    e = zero(T)

    ŵ = sum(w)
    for i in eachindex(Q,P)
        e += norm(Q[i] - R*P[i] - t)*w[i]
    end
    return e/ŵ
end

function metricerror(
    Q::AbstractMaybePoints{T},
    P::AbstractPoints{T},
    w::AbstractVector{T},
    t::SVector{3,T},
    idxs,
    R::Rotation{3,T}
) where T
    e = zero(T)

    ŵ = sum(w)
    for i in idxs
        e += norm(SVector{3,T}(Q[i]) - R*P[i] - t)*w[i]
    end
    return e/ŵ
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
