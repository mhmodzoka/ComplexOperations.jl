using ComplexOperations
using LinearAlgebra

# testing complex_matrix_inversion
A = rand(3, 3); B = rand(3, 3)
@time for i = 1:1e6; C, D = complex_matrix_inversion(A, B); end
@time for i = 1:1e6; Z = inv(A + im * B); end
C, D = complex_matrix_inversion(A, B)
Z = inv(A + im * B)
Z == complex.(C, D) # TODO: why there is a small difference?



# testing complex_vector_dot_product
A_r = [1,2,3]; A_i = [1, 20, 30];
B_r = [10,2,30]; B_i = [1, 2, 3];
println(); println("A and B are complex")
@time AB = complex_vector_dot_product(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_dot_product(hcat(A_r, A_i), hcat(B_r, B_i))

println(); println("A is real, B is complex")
@time AB__ = complex_vector_dot_product(hcat(A_r, zero(A_r)), hcat(B_r, B_i))
@time AB__ = complex_vector_dot_product(A_r, hcat(B_r, B_i))

println(); println("A is complex, B is real")
@time AB_ = complex_vector_dot_product(hcat(A_r, A_i), hcat(B_r, zero(B_r)))
@time AB_ = complex_vector_dot_product(hcat(A_r, A_i), B_r)


# testing complex_vector_dot_product, returns SMatrix
A_r = [1,2,3]; A_i = [1, 20, 30];
B_r = [10,2,30]; B_i = [1, 2, 3];
println(); println("A and B are complex")
@time AB = complex_vector_dot_product_SMatrix(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_dot_product_SMatrix(hcat(A_r, A_i), hcat(B_r, B_i))

println(); println("A is real, B is complex")
@time AB__ = complex_vector_dot_product_SMatrix(hcat(A_r, zero(A_r)), hcat(B_r, B_i))
@time AB__ = complex_vector_dot_product_SMatrix(A_r, hcat(B_r, B_i))

println(); println("A is complex, B is real")
@time AB_ = complex_vector_dot_product_SMatrix(hcat(A_r, A_i), hcat(B_r, zero(B_r)))
@time AB_ = complex_vector_dot_product_SMatrix(hcat(A_r, A_i), B_r)


# testing complex_vector_cross_product
A_r = [1,2,3]; A_i = [1, 20, 30];
B_r = [10,2,30]; B_i = [1, 2, 3];
println(); println("A and B are complex")
@time AB = complex_vector_cross_product(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_cross_product(hcat(A_r, A_i), hcat(B_r, B_i))

println(); println("A is real, B is complex")
@time AB__ = complex_vector_cross_product(hcat(A_r, zero(A_r)), hcat(B_r, B_i))
@time AB__ = complex_vector_cross_product(A_r, hcat(B_r, B_i))

println(); println("A is complex, B is real")
@time AB_ = complex_vector_cross_product(hcat(A_r, A_i), hcat(B_r, zero(B_r)))
@time AB_ = complex_vector_cross_product(hcat(A_r, A_i), B_r)


# testing complex_vector_cross_product, returns SMatrix
A_r = [1,2,3]; A_i = [1, 20, 30];
B_r = [10,2,30]; B_i = [1, 2, 3];
println(); println("A and B are complex")
@time AB = complex_vector_cross_product_SMatrix(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_cross_product_SMatrix(hcat(A_r, A_i), hcat(B_r, B_i))

println(); println("A is real, B is complex")
@time AB__ = complex_vector_cross_product_SMatrix(hcat(A_r, zero(A_r)), hcat(B_r, B_i))
@time AB__ = complex_vector_cross_product_SMatrix(A_r, hcat(B_r, B_i))

println(); println("A is complex, B is real")
@time AB_ = complex_vector_cross_product_SMatrix(hcat(A_r, A_i), hcat(B_r, zero(B_r)))
@time AB_ = complex_vector_cross_product_SMatrix(hcat(A_r, A_i), B_r)




#################### which approach is best for complex matrix inversion?
function complex_matrix_inversion_1(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    @fastmath @inbounds inv_A_B = inv(A) * B
    @fastmath @inbounds C = inv(A + B * inv_A_B)
    @fastmath @inbounds D = -inv_A_B * C
    return C, D
end

function complex_matrix_inversion_2(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real
    @fastmath @inbounds inv_A_B = inv(A) * B
    return ((inv(A + B * inv_A_B)), -inv_A_B * (inv(A + B * inv_A_B)))
end

function complex_matrix_inversion_3(A::AbstractMatrix{R}, B::AbstractMatrix{R}) where R <: Real    
    return ((inv(A + B * inv(A) * B)), -inv(A) * B * (inv(A + B * inv(A) * B)))
end

dim_ = 5
@time for i = 1:1e4; A, B = rand(dim_, dim_), rand(dim_, dim_); complex_matrix_inversion_1(A, B); end
@time for i = 1:1e4; A, B = rand(dim_, dim_), rand(dim_, dim_); complex_matrix_inversion_2(A, B); end
@time for i = 1:1e4; A, B = rand(dim_, dim_), rand(dim_, dim_); complex_matrix_inversion_3(A, B); end