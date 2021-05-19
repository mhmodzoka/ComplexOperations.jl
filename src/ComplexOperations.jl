module ComplexOperations

export complex_multiply
export complex_divide
export complex_vector_dot_product
export complex_vector_cross_product
export complex_matrix_inversion
export complex_matrix_multiplication

"""
    Multiplication of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(Z1, Z2)
    return vcat(complex_multiply.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end
"""
    Multiplication of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all real numbers or arrays of real numbers. All shoud have the same size.
Multiplication algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_multiply(a, b, c, d)
    # return hcat(a * c - b * d, b * c + a * d)
    return @fastmath @inbounds hcat(a .* c - b .* d, b .* c + a .* d)
end

"""
    Division of two vectors Z1 and Z2.
We assume the first and second columns of each vector are the real and imaginary parts, respectively
Each of Z1 and Z2 has two columns, and arbitrary number of rows.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(Z1, Z2)
    return vcat(complex_divide.(Z1[:,1], Z1[:,2], Z2[:,1], Z2[:,2])...)
end
"""
    Division of two vectors Z1=a+im*b and Z2=c+im*d.
a, b, c, and d are all real numbers or arrays of real numbers. All shoud have the same size.
Division algorithm for complex numbers can be found here: https://en.wikipedia.org/wiki/Multiplication_algorithm
"""
function complex_divide(a, b, c, d)    
    return @fastmath @inbounds hcat((a .* c + b .* d) / (c.^2 + d.^2), (b .* c - a .* d) / (c.^2 + d.^2))
end

"""
    Claculate dot product of two vectors
Each is represented as 3x2 or 3x1 Array. The first and second columns represent the real and imag parts, respectively
For real vectors, we can input them as 3x1 Array or 3-element Vector
"""
function complex_vector_dot_product(A, B)
    # TODO: how can I create two methods, rather than the if-statements?
    if size(A, 2) == 1; A = hcat(A, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    if size(B, 2) == 1; B = hcat(B, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    return complex_vector_dot_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate dot product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_dot_product(A_r, A_i, B_r, B_i)
    return (
    complex_multiply(A_r[1], A_i[1], B_r[1], B_i[1]) +
    complex_multiply(A_r[2], A_i[2], B_r[2], B_i[2]) +
    complex_multiply(A_r[3], A_i[3], B_r[3], B_i[3])
)  
end

"""
    Claculate cross product of two vectors
Each is represented as 3x2 or 3x1 Array. The first and second columns represent the real and imag parts, respectively
For real vectors, we can input them as 3x1 Array or 3-element Vector
"""
function complex_vector_cross_product(A, B)
    if size(A, 2) == 1; A = hcat(A, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    if size(B, 2) == 1; B = hcat(B, [0,0,0]); end # TODO: is there a better way to handle 3x2 and 3x1 vectors?
    return complex_vector_cross_product(A[:,1], A[:,2], B[:,1], B[:,2])
end

"""
    Claculate cross product of two vectors, inputs are real and imaginary parts of the two vectors
Each is represented as 3-element array.
"""
function complex_vector_cross_product(A_r, A_i, B_r, B_i)
    vcat(
    complex_multiply(A_r[2], A_i[2], B_r[3], B_i[3]) - complex_multiply(A_r[3], A_i[3], B_r[2], B_i[2]),
    complex_multiply(A_r[3], A_i[3], B_r[1], B_i[1]) - complex_multiply(A_r[1], A_i[1], B_r[3], B_i[3]),
    complex_multiply(A_r[1], A_i[1], B_r[2], B_i[2]) - complex_multiply(A_r[2], A_i[2], B_r[1], B_i[1]),
)
end

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
function complex_matrix_inversion(A, B)    
    @fastmath @inbounds inv_A_B = inv(A) * B
    @fastmath @inbounds C = inv(A + B * inv_A_B)
    @fastmath @inbounds D = -inv_A_B * C
    return C, D
end

"""
    Multiply two matrices A+iB and C+iD, where A,B,C,D are real matrices
"""
function complex_matrix_multiplication(A, B, C, D)
    return @fastmath @inbounds (A * C - B * D, A * D + B * C)
end


end # module
