# the following functions must be implemented by the package defining the vector type (e.g. SIMD.jl)

"""
    T = precision(x)
Returns either `Float32` or `Float64` for SIMD object `x`. Used to select an appropriate implementation of special functions.
""" 
function precision end

"""
    x = vifelse(condition, x_if_true, x_if_false)
As `Base.ifelse`, for vector arguments.
"""
function vifelse end

"""
    y = unsafe_trunc(T, x)
As `Base.unsafe_trunc`, for vector arguments.
"""
function vunsafe_trunc end

# the following functions are implemented by the package providing math functions (e.g. SLEEF.jl)

"""
    vexp(x)

Compute the base-`e` exponential of `x`, that is `eˣ`.
"""
function vexp end
