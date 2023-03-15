"""
    vexp(x)

Compute the base-`e` exponential of `x`, that is `eˣ`.
"""
@inline vexp(x) = vexp(precision(x), x)

# element_type(x::Union{Float16, Float32, Float64}) = typeof(x)

@inline function vexp(::Type{T}, d) where {T<:Union{Float32,Float64}}
    q = round(T(MLN2E) * d)
    qi = vunsafe_trunc(int_type(T), q)

    s = muladd(q, -L2U(T), d)
    s = muladd(q, -L2L(T), s)

    u = vexp_kernel(T, s)

    u = muladd(s * s, u, s + one(u))
    u = vldexp2k(u, qi)
    
    u = vifelse(d > max_exp(T), T(Inf), u)
    u = vifelse(d < min_exp(T), T(0), u)

    return u
end

int_type(::Type{Float64}) = Int64
int_type(::Type{Float32}) = Int32

@inline function vexp_kernel(::Type{Float64}, x)
    c11 = 2.08860621107283687536341e-09
    c10 = 2.51112930892876518610661e-08
    c9  = 2.75573911234900471893338e-07
    c8  = 2.75572362911928827629423e-06
    c7  = 2.4801587159235472998791e-05
    c6  = 0.000198412698960509205564975
    c5  = 0.00138888888889774492207962
    c4  = 0.00833333333331652721664984
    c3  = 0.0416666666666665047591422
    c2  = 0.166666666666666851703837
    c1  = 0.50
    return @horner x c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11
end

@inline function vexp_kernel(::Type{Float32}, x)
    c6 = 0.000198527617612853646278381f0
    c5 = 0.00139304355252534151077271f0
    c4 = 0.00833336077630519866943359f0
    c3 = 0.0416664853692054748535156f0
    c2 = 0.166666671633720397949219f0
    c1 = 0.5f0
    return @horner x*x muladd(x,c2,c1) muladd(x,c4,c3) muladd(x,c6,c5)
end

# adapted from priv.jl

@inline function vldexp2k(x, e)
    P, T = precision(x), typeof(x)
    return x * vpow2i(P, T, e >> 1) * vpow2i(P, T, e - (e >> 1))
end

# adapted from utils.jl

@inline vpow2i(::Type{P}, ::Type{T}, q) where {P,T} = vinteger2float(P, T, q + exponent_bias(P))

@inline vinteger2float(::Type{Float64}, ::Type{T}, m) where T = reinterpret(T, m << significand_bits(Float64))
@inline vinteger2float(::Type{Float32}, ::Type{T}, m) where T = reinterpret(T, m << significand_bits(Float32))
