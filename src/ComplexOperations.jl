module ComplexOperations

using StaticArrays

export complex_multiply
export complex_multiply_SMatrix

export complex_divide
export complex_divide_SMatrix

export complex_vector_dot_product
export complex_vector_dot_product_SMatrix

export complex_vector_cross_product
export complex_vector_cross_product_SMatrix

export complex_matrix_inversion
export complex_matrix_multiplication

# Scalar multiplication -----------------------------------------------------------------------------------
"""
    Multiplication of two complex numbers Z1=a+im*b and Z2=c+im*d, defined by a,b,c and d.
a, b, c, and d are all real numbers.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(a::R, b::R, c::R, d::R) where R <: Real
    return @fastmath @inbounds hcat(a * c - b * d, b * c + a * d)
end

"""
    Multiplication of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all arrays of real numbers, all arrays have the same size.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(a::AbstractVecOrMat{R}, b::AbstractVecOrMat{R}, c::AbstractVecOrMat{R}, d::AbstractVecOrMat{R}) where R <: Real       
    return @fastmath @inbounds hcat(a .* c - b .* d, b .* c + a .* d)
end

"""
    Multiplication of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(Z1::AbstractVecOrMat{R}, Z2::AbstractVecOrMat{R}) where R <: Real
    return vcat(complex_multiply.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end

# Scalar multiplication, returns SMatrix -----------------------------------------------------------------------------------
"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(a::R, b::R, c::R, d::R) where R <: Real
    return @fastmath @inbounds SMatrix{1,2}(a * c - b * d, b * c + a * d)
end


# TODO: remove this one, it is confusing
"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(a::AbstractVecOrMat{R}, b::AbstractVecOrMat{R}, c::AbstractVecOrMat{R}, d::AbstractVecOrMat{R}) where R <: Real       
    return @fastmath @inbounds vcat(complex_multiply_SMatrix.(a, b, c, d)...)
end

"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(Z1::AbstractVecOrMat{R}, Z2::AbstractVecOrMat{R}) where R <: Real
    return vcat(complex_multiply_SMatrix.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end









# Scalar division -----------------------------------------------------------------------------------
"""
    Division of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all real numbers
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(a::R, b::R, c::R, d::R) where R <: Real
    return @fastmath @inbounds hcat((a * c + b * d) / (c^2 + d^2), (b * c - a * d) / (c^2 + d^2))
end

"""
    Division of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all arrays of real numbers, all arrays have the same size.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(a::AbstractVecOrMat{R}, b::AbstractVecOrMat{R}, c::AbstractVecOrMat{R}, d::AbstractVecOrMat{R}) where R <: Real
    return @fastmath @inbounds hcat((a .* c + b .* d) / (c.^2 + d.^2), (b .* c - a .* d) / (c.^2 + d.^2))
end

"""
    Division of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(Z1::AbstractVecOrMat{R}, Z2::AbstractVecOrMat{R}) where R <: Real
    return vcat(complex_divide.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end

# Scalar division, returns SMatrix -----------------------------------------------------------------------------------
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(a::R, b::R, c::R, d::R) where R <: Real
    return @fastmath @inbounds SMatrix{1,2}((a * c + b * d) / (c^2 + d^2), (b * c - a * d) / (c^2 + d^2))
end

# TODO: remove this one, it is confusing
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(a::AbstractVecOrMat{R}, b::AbstractVecOrMat{R}, c::AbstractVecOrMat{R}, d::AbstractVecOrMat{R}) where R <: Real
    return @fastmath @inbounds vcat(complex_divide_SMatrix.(a, b, c, d)...)
end
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(Z1::AbstractVecOrMat{R}, Z2::AbstractVecOrMat{R}) where R <: Real
    return vcat(complex_divide_SMatrix.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end


# Vector dot product ------------------------------------------------------------------------------------------
"""
    Claculate dot product of two vectors
Inputs are real and imaginary parts of the two vectors, each is represented as 3-element array.
"""
function complex_vector_dot_product(A_r::AbstractVector{R}, A_i::AbstractVector{R}, B_r::AbstractVector{R}, B_i::AbstractVector{R}) where R <: Real
    return (
    complex_multiply(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate dot product of two vectors. A and B represents complex vectors.
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_dot_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent real and complex vectors, respectively.
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractVector{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_dot_product(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent complex and real vectors, respectively.
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractMatrix{R}, B::AbstractVector{R}) where R <: Real
    return complex_vector_dot_product(A[:,1], A[:,2], B, zero(B))
end



# Vector dot product, returns SMatrix ------------------------------------------------------------------------------------------
"""
    Claculate dot product of two vectors, and returns SMatrix.
Inputs are real and imaginary parts of the two vectors, each is represented as 3-element array.
"""
function complex_vector_dot_product_SMatrix(A_r::AbstractVector{R}, A_i::AbstractVector{R}, B_r::AbstractVector{R}, B_i::AbstractVector{R}) where R <: Real
    return (
    complex_multiply_SMatrix(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply_SMatrix(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply_SMatrix(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate dot product of two vectors. A and B represents complex vectors. Returns SMatrix
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_dot_product_SMatrix(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent real and complex vectors, respectively. Returns SMatrix
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractVector{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_dot_product_SMatrix(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent complex and real vectors, respectively. Returns SMatrix
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractMatrix{R}, B::AbstractVector{R}) where R <: Real
    return complex_vector_dot_product_SMatrix(A[:,1], A[:,2], B, zero(B))
end




# Vector cross product ------------------------------------------------------------------------------------------
"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_cross_product(A_r::AbstractVector{R}, A_i::AbstractVector{R}, B_r::AbstractVector{R}, B_i::AbstractVector{R}) where R <: Real
    return vcat(
    complex_multiply(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

"""
    Claculate cross product of two vectors. A and B represents complex vectors.
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_cross_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent real and complex vectors, respectively.
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractVector{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_cross_product(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent complex and real vectors, respectively.
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractMatrix{R}, B::AbstractVector{R}) where R <: Real
    return complex_vector_cross_product(A[:,1], A[:,2], B, zero(B))
end


# Vector cross product, returns SMatrix ------------------------------------------------------------------------------------------
"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors. Returns SMatrix
Each is represented as 3-element array.
"""
function complex_vector_cross_product_SMatrix(A_r::AbstractVector{R}, A_i::AbstractVector{R}, B_r::AbstractVector{R}, B_i::AbstractVector{R}) where R <: Real
    return vcat(
    complex_multiply_SMatrix(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply_SMatrix(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply_SMatrix(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply_SMatrix(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply_SMatrix(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply_SMatrix(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

"""
    Claculate cross product of two vectors. A and B represents complex vectors. Returns SMatrix
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_cross_product_SMatrix(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent real and complex vectors, respectively. Returns SMatrix
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractVector{R}, B::AbstractMatrix{R}) where R <: Real
    return complex_vector_cross_product_SMatrix(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent complex and real vectors, respectively. Returns SMatrix
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractMatrix{R}, B::AbstractVector{R}) where R <: Real
    return complex_vector_cross_product_SMatrix(A[:,1], A[:,2], B, zero(B))
end





# Complex matrix inverse ------------------------------------------------------------------------------------------
"""
    Calculate the inverse of a complex matrix A+iB, where A and B are real matrices

There are two possible versions:
Version #1: This one preallocate. Faster and use more memory
for 1M evaluations: 1.771474 seconds (14.18 M allocations: 4.510 GiB, 4.52% gc time, 6.40% compilation time)
`
function complex_matrix_inversion(A, B)
    inv_A_B = inv(A) * B
    C = inv(A + B * inv_A_B)
    D = -inv_A_B * C
    return C, D
end
`

Version #2: No preallocation. Slower and use less memory
for 1M evaluations: 2.330454 seconds (19.00 M allocations: 6.512 GiB, 2.71% gc time)
`
function complex_matrix_inversion(A, B)    
    C = inv(A + B * inv(A) * B)
    D = -1 .* inv(A) * B * C
    return C, D
end
`
"""
function complex_matrix_inversion(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    @fastmath @inbounds inv_A_B = inv(A) * B
    @fastmath @inbounds C = inv(A + B * inv_A_B)
    @fastmath @inbounds D = -inv_A_B * C
    return C, D
end


# Complex matrix multiplication ------------------------------------------------------------------------------------------
"""
    Multiply two matrices A+iB and C+iD, where A,B,C,D are real matrices
"""
function complex_matrix_multiplication(A::AbstractMatrix{R}, B::AbstractMatrix{R}, C::AbstractMatrix{R}, D::AbstractMatrix{R}) where R <: Real
    return @fastmath @inbounds (A * C - B * D, A * D + B * C)
end



##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
# TODO: find a better way to accomodate different types.
# in what follows, I am adding more methods for all
# Scalar multiplication -----------------------------------------------------------------------------------
"""
    Multiplication of two complex numbers Z1=a+im*b and Z2=c+im*d, defined by a,b,c and d.
a, b, c, and d are all real numbers.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(a::Real, b::Real, c::Real, d::Real)
    return @fastmath @inbounds hcat(a * c - b * d, b * c + a * d)
end

"""
    Multiplication of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all arrays of real numbers, all arrays have the same size.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(a::AbstractVecOrMat{Real}, b::AbstractVecOrMat{Real}, c::AbstractVecOrMat{Real}, d::AbstractVecOrMat{Real})       
    return @fastmath @inbounds hcat(a .* c - b .* d, b .* c + a .* d)
end

"""
    Multiplication of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(Z1::AbstractVecOrMat{Real}, Z2::AbstractVecOrMat{Real})
    return vcat(complex_multiply.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end

# Scalar multiplication, returns SMatrix -----------------------------------------------------------------------------------
"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(a::Real, b::Real, c::Real, d::Real)
    return @fastmath @inbounds SMatrix{1,2}(a * c - b * d, b * c + a * d)
end


# TODO: remove this one, it is confusing
"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(a::AbstractVecOrMat{Real}, b::AbstractVecOrMat{Real}, c::AbstractVecOrMat{Real}, d::AbstractVecOrMat{Real})       
    return @fastmath @inbounds vcat(complex_multiply_SMatrix.(a, b, c, d)...)
end

"""
    Same as `complex_multiply`, but return SMatrix
"""
function complex_multiply_SMatrix(Z1::AbstractVecOrMat{Real}, Z2::AbstractVecOrMat{Real})
    return vcat(complex_multiply_SMatrix.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end









# Scalar division -----------------------------------------------------------------------------------
"""
    Division of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all real numbers
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(a::Real, b::Real, c::Real, d::Real)
    return @fastmath @inbounds hcat((a * c + b * d) / (c^2 + d^2), (b * c - a * d) / (c^2 + d^2))
end

"""
    Division of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all arrays of real numbers, all arrays have the same size.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(a::AbstractVecOrMat{Real}, b::AbstractVecOrMat{Real}, c::AbstractVecOrMat{Real}, d::AbstractVecOrMat{Real})
    return @fastmath @inbounds hcat((a .* c + b .* d) / (c.^2 + d.^2), (b .* c - a .* d) / (c.^2 + d.^2))
end

"""
    Division of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(Z1::AbstractVecOrMat{Real}, Z2::AbstractVecOrMat{Real})
    return vcat(complex_divide.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end

# Scalar division, returns SMatrix -----------------------------------------------------------------------------------
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(a::Real, b::Real, c::Real, d::Real)
    return @fastmath @inbounds SMatrix{1,2}((a * c + b * d) / (c^2 + d^2), (b * c - a * d) / (c^2 + d^2))
end

# TODO: remove this one, it is confusing
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(a::AbstractVecOrMat{Real}, b::AbstractVecOrMat{Real}, c::AbstractVecOrMat{Real}, d::AbstractVecOrMat{Real})
    return @fastmath @inbounds vcat(complex_divide_SMatrix.(a, b, c, d)...)
end
"""
    Same as `complex_divide`, but returns SMatrix
"""
function complex_divide_SMatrix(Z1::AbstractVecOrMat{Real}, Z2::AbstractVecOrMat{Real})
    return vcat(complex_divide_SMatrix.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end


# Vector dot product ------------------------------------------------------------------------------------------
"""
    Claculate dot product of two vectors
Inputs are real and imaginary parts of the two vectors, each is represented as 3-element array.
"""
function complex_vector_dot_product(A_r::AbstractVector{Real}, A_i::AbstractVector{Real}, B_r::AbstractVector{Real}, B_i::AbstractVector{Real})
    return (
    complex_multiply(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate dot product of two vectors. A and B represents complex vectors.
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractMatrix{Real}, B::AbstractMatrix{Real})
    return complex_vector_dot_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent real and complex vectors, respectively.
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractVector{Real}, B::AbstractMatrix{Real})
    return complex_vector_dot_product(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent complex and real vectors, respectively.
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product(A::AbstractMatrix{Real}, B::AbstractVector{Real})
    return complex_vector_dot_product(A[:,1], A[:,2], B, zero(B))
end



# Vector dot product, returns SMatrix ------------------------------------------------------------------------------------------
"""
    Claculate dot product of two vectors, and returns SMatrix.
Inputs are real and imaginary parts of the two vectors, each is represented as 3-element array.
"""
function complex_vector_dot_product_SMatrix(A_r::AbstractVector{Real}, A_i::AbstractVector{Real}, B_r::AbstractVector{Real}, B_i::AbstractVector{Real})
    return (
    complex_multiply_SMatrix(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply_SMatrix(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply_SMatrix(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate dot product of two vectors. A and B represents complex vectors. Returns SMatrix
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractMatrix{Real}, B::AbstractMatrix{Real})
    return complex_vector_dot_product_SMatrix(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent real and complex vectors, respectively. Returns SMatrix
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractVector{Real}, B::AbstractMatrix{Real})
    return complex_vector_dot_product_SMatrix(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors. A and B represent complex and real vectors, respectively. Returns SMatrix
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_dot_product_SMatrix(A::AbstractMatrix{Real}, B::AbstractVector{Real})
    return complex_vector_dot_product_SMatrix(A[:,1], A[:,2], B, zero(B))
end




# Vector cross product ------------------------------------------------------------------------------------------
"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_cross_product(A_r::AbstractVector{Real}, A_i::AbstractVector{Real}, B_r::AbstractVector{Real}, B_i::AbstractVector{Real})
    return vcat(
    complex_multiply(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

"""
    Claculate cross product of two vectors. A and B represents complex vectors.
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractMatrix{Real}, B::AbstractMatrix{Real})
    return complex_vector_cross_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent real and complex vectors, respectively.
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractVector{Real}, B::AbstractMatrix{Real})
    return complex_vector_cross_product(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent complex and real vectors, respectively.
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product(A::AbstractMatrix{Real}, B::AbstractVector{Real})
    return complex_vector_cross_product(A[:,1], A[:,2], B, zero(B))
end


# Vector cross product, returns SMatrix ------------------------------------------------------------------------------------------
"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors. Returns SMatrix
Each is represented as 3-element array.
"""
function complex_vector_cross_product_SMatrix(A_r::AbstractVector{Real}, A_i::AbstractVector{Real}, B_r::AbstractVector{Real}, B_i::AbstractVector{Real})
    return vcat(
    complex_multiply_SMatrix(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply_SMatrix(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply_SMatrix(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply_SMatrix(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply_SMatrix(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply_SMatrix(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

"""
    Claculate cross product of two vectors. A and B represents complex vectors. Returns SMatrix
A represents a complex vector and is represented as 3x2 Matrix, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractMatrix{Real}, B::AbstractMatrix{Real})
    return complex_vector_cross_product_SMatrix(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent real and complex vectors, respectively. Returns SMatrix
A represents a real vector and is represented as 3-element vector, while B represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractVector{Real}, B::AbstractMatrix{Real})
    return complex_vector_cross_product_SMatrix(A, zero(A), B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors. A and B represent complex and real vectors, respectively. Returns SMatrix
B represents a real vector and is represented as 3-element vector, while A represents a complex vector and is represented as 3x2 Matrix
"""
function complex_vector_cross_product_SMatrix(A::AbstractMatrix{Real}, B::AbstractVector{Real})
    return complex_vector_cross_product_SMatrix(A[:,1], A[:,2], B, zero(B))
end





# Complex matrix inverse ------------------------------------------------------------------------------------------
"""
    Calculate the inverse of a complex matrix A+iB, where A and B are real matrices

There are two possible versions:
Version #1: This one preallocate. Faster and use more memory
for 1M evaluations: 1.771474 seconds (14.18 M allocations: 4.510 GiB, 4.52% gc time, 6.40% compilation time)
`
function complex_matrix_inversion(A, B)
    inv_A_B = inv(A) * B
    C = inv(A + B * inv_A_B)
    D = -inv_A_B * C
    return C, D
end
`

Version #2: No preallocation. Slower and use less memory
for 1M evaluations: 2.330454 seconds (19.00 M allocations: 6.512 GiB, 2.71% gc time)
`
function complex_matrix_inversion(A, B)    
    C = inv(A + B * inv(A) * B)
    D = -1 .* inv(A) * B * C
    return C, D
end
`
"""
function complex_matrix_inversion(A::AbstractMatrix{Real}, B::AbstractMatrix{Real})
    @fastmath @inbounds inv_A_B = inv(A) * B
    @fastmath @inbounds C = inv(A + B * inv_A_B)
    @fastmath @inbounds D = -inv_A_B * C
    return C, D
end


# Complex matrix multiplication ------------------------------------------------------------------------------------------
"""
    Multiply two matrices A+iB and C+iD, where A,B,C,D are real matrices
"""
function complex_matrix_multiplication(A::AbstractMatrix{Real}, B::AbstractMatrix{Real}, C::AbstractMatrix{Real}, D::AbstractMatrix{Real})
    return @fastmath @inbounds (A * C - B * D, A * D + B * C)
end



end # module
