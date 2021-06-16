# ComplexOperations.jl

Perform basic operations on complex numbers. Complex numbers are represented as 2-element vectors. Functions here can return StaticArrays for extra speed.
Operations include:

- complex scalar operations
  - scalar multiplications (`complex_multiply`, `complex_multiply_SMatrix`)
  - scalar division (`complex_divide`, `complex_divide_SMatrix`)
- complex vector operations
  - vector dot product (`complex_vector_dot_product`, `complex_vector_dot_product_SMatrix`)
  - vector cross product (`complex_vector_cross_product`, `complex_vector_cross_product_SMatrix`)
- complex matrix operations
  - matrix multiplication (`complex_matrix_multiplication`)
  - matrix inversion (`complex_matrix_inversion`)

# Installation

The package is registered, and can be installed by executing the following command:

`using Pkg; Pkg.add("ComplexOperations")`

# How does it work?
Check `docs`

