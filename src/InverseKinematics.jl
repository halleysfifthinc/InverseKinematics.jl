module InverseKinematics

using LinearAlgebra,
      Statistics

using LinearAlgebra: norm_sqr

using Rotations, CoordinateTransformations, StaticArrays

export kabsch,
       fa3r,
       fitseg,
       metricerror,
       rmat,
       unit

export SegOpt,
       Kabsch,
       FA3R

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
    pos::SVector{3,T}
    ori::Quat{T}
    prox::SVector{3,T}
    dist::SVector{3,T}
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

function fitseg(::Kabsch, Q, P)
    kabsch(Q, P)
end

function fitseg(f::FA3R, Q, P)
    fa3r(Q, P, f.maxiters, f.ϵ)
end

function kabsch(Q::AbstractMaybePoints{T}, P::AbstractPoints{T}) where T
    present = findpresent(Q)

    _kabsch(Q, P, present)
end

function kabsch(Q::AbstractPoints{T}, P::AbstractPoints{T}) where T
    present = axes(Q, 1)

    _kabsch(Q, P, present)
end

function _kabsch(Q::AbstractMaybePoints{T}, P::AbstractPoints{T}, present::MVector) where T
    S = length(present)
    if S < 3 # A full fit isn't possible
        return incompletefit(S, Q, P, present)
    end
    _Q = Q[present]
    _P = P[present]

    # Calculate point set's centroid
    q_cent, p_cent = mean(_Q), mean(P)

    H = zero(SArray{Tuple{3,3},T})
    for i in eachindex(_Q, _P)
        p′ = SArray{Tuple{3,1},T}(_P[i] - p_cent)
        q′ = SArray{Tuple{1,3},T}(_Q[i] - q_cent)

        # Cross-covariance matrix
        H = H + (p′ * q′)
    end

    # SVD
    Hsvd = svd(H)
    V = Hsvd.V
    U = Hsvd.U

    # Correct for handedness (corrects to a right-handed system)
    Hp = V*U'
    di = SMatrix{3,3}(Diagonal(SVector{3}(1, 1, sign(det(Hp)))))

    R = V * di * U'
    t = q_cent - p_cent
    e = metricerror(Q, P, t, present, R)

    qt = Quat(R)

    return (qt, t, e)
end

function fa3r(Q::AbstractMaybePoints{T}, P::AbstractPoints{T}, maxiters::Int=100, ϵ::T=0) where T
    present = findpresent(Q)

    _fa3r(Q, P, present, maxiters, ϵ)
end

function fa3r(Q::AbstractPoints{T}, P::AbstractPoints{T}, maxiters::Int=100, ϵ::T=0) where T
    present = axes(Q, 1)

    _fa3r(Q, P, present, maxiters, ϵ)
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

function incompletefit(S::Int, Q::AbstractMaybePoints{T}, P::AbstractPoints{T}, present::MVector) where T
    t = zero(SVector{3, T})

    if 0 < S # We can get a very rough position
        q_cent, p_cent = mean(SVector{S,SVector{3,T}}(Q[present])), mean(P)
        t = q_cent - p_cent
    end

    qt = one(Quat{T})
    e = T(NaN)

    return (qt, t, e)
end


function _fa3r(Q::AbstractMaybePoints{T}, P::AbstractPoints{T}, present::MVector, maxiters::Int, ϵ::T)::Tuple{Quat{T},SVector{3,T},T} where T
    S = length(present)
    if S < 3 # A full fit isn't possible
        return incompletefit(S, Q, P, present)
    end

    #Calculate point set's centroid
    q_cent, p_cent = mean(SVector{S,SVector{3,T}}(Q[present])), mean(P)

    H = zero(SArray{Tuple{3,3},T})
    for i in present
        p′ = SArray{Tuple{3,1},T}(P[i] - p_cent)
        q′ = SArray{Tuple{1,3},T}(SVector{3,T}(Q[i]) - q_cent)

        # Cross-covariance matrix
        H = H + (p′ * q′)
    end

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
    t = q_cent - p_cent
    e = metricerror(Q, P, t, present, R)

    qt = Quat(R)

    return (qt, t, e)
end

function metricerror(Q::AbstractPoints{T}, P::AbstractPoints{T}, t::SVector{3,T}, R::SMatrix{3,3,T}) where T
    e = zero(eltype(R))
    n = length(Q)

    for i in eachindex(Q,P)
        e += norm(Q[i] - R*P[i] - t)
    end
    return e/n
end

function metricerror(Q::AbstractMaybePoints{T}, P::AbstractPoints{T}, t::SVector{3,T}, idxs, R::SMatrix{3,3,T}) where T
    e = zero(eltype(R))
    n = length(idxs)

    for i in idxs
        e += norm(SVector{3,T}(Q[i]) - R*P[i] - t)
    end
    return e/n
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
