using .SIMD: SIMD # do not import anything beyond SIMD itself

# VectorAPI functions that should be provided by SIMD.jl

precision(::SIMD.Vec{N,T}) where {N,T}=T
vunsafe_trunc(::Type{I}, x::SIMD.Vec{N,T}) where {I,N,T} = SIMD.Vec(SIMD.Intrinsics.fptosi(SIMD.LVec{N,I}, x.data))
vifelse(a::SIMD.Vec, b, c) = SIMD.vifelse(a,b,c)

# Use VectorAPI for special functions with SIMD.Vec argument
# To be moved to SIMD.jl or a future SIMDFunctions.jl ( SIMDFunctions.jl => SIMD.jl => VectorAPI.jl)

Base.exp(x::SIMD.Vec)               = vexp(x)
Base.FastMath.exp_fast(x::SIMD.Vec) = vexp(x)
