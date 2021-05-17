module ComplexOperations

export complex_multiply
export complex_divide

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
    return hcat(a .* c - b .* d, b .* c + a .* d)
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
    return hcat((a .* c + b .* d) / (c.^2 + d.^2), (b .* c - a .* d) / (c.^2 + d.^2))
end
end # module
