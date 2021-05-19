using ComplexOperations
using LinearAlgebra

# testing complex_matrix_inversion
A = rand(3, 3); B = rand(3, 3)
@time for i = 1:1e6; C, D = complex_matrix_inversion(A, B); end
@time for i = 1:1e6; Z = inv(A + im * B); end
C, D = complex_matrix_inversion(A, B)
Z = inv(A + im * B)
Z == complex.(C, D) # TODO: why there is a small difference?

# testing complex_vector_cross_product
A_r = [1,2,3]; A_i = [1, 20, 30]
B_r = [10,2,30]; B_i = [1, 2, 3]
@time AB = complex_vector_cross_product(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_cross_product(hcat(A_r, A_i), hcat(B_r, B_i))
@time AB__ = cross(A_r + im * A_i, B_r + im * B_i)

# testing complex_vector_dot_product
A_r = [1,2,3]; A_i = [1, 20, 30]
B_r = [10,2,30]; B_i = [1, 2, 3]
@time AB = complex_vector_dot_product(A_r, A_i, B_r, B_i)
@time AB_ = complex_vector_dot_product(hcat(A_r, A_i), hcat(B_r, B_i))
@time AB__ = Tmatrix.vector_dot_product(A_r + im * A_i, B_r + im * B_i)