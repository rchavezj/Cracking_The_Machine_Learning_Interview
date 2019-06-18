# Linear Algebra
[(Return back to Contents)](https://github.com/rchavezj/Cracking_The_Machine_Learning_Interview/#Contents)

<img src="linear_algebra.png" width="700">

### 1. What is broadcasting in connection to Linear Algebra?
### 2. What are scalars, vectors, matrices, and tensors?
Vectors have three different perspective: physics, computer science (machine learning), and mathamatics. The physics student perspective is that vectors are arrows pointing in space. What defines a given vector is it's length, and the direction it's pointing in, but as long as those two facts are the same, you can move it all around and it's still the same vector. Vectors that live in the flat plane are 2d, and those sitting in broader space that you and I live in are 3d. The computer scientist perspective is that vectors are ordered lists of numbers. For example, lets that you were doing some analytics about house prices, and the only features you cared about were square footage and price. You might model each house with a pair of numbers: the first indicating sqaure footsage, and the second indicating price. In the machine learning community, each row from one column represents the number of training examples. The mathmatician tries to generalize both of the physicist and computer scientist. </br> <img src="physics.png" width="288" height="200"><img src="math_vector.png" width="288" height="200"><img src="cs_vector.png" width="288" height="200">
Scalers are numerical values to stretch, squeeze, or even change to the opposite direction of a given vector. Scalers are computed through multiplication. 
   
   <center><img src="scaling_v2.png" width="300"></center>

Matrices are similar to vectors except each column is another set of features for the dataset.</br>
</br><center><img src="matrices.png" width="500"></center>
### 3. What is Hadamard product of two matrices?
### 4. What is an inverse matrix?
### 5. If inverse of a matrix exists, how to calculate it?
### 6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?
> <img src="determinant.png">
### 7. Discuss span and linear dependence.
### 8. Following up on question #7, what does the following definition mean, "The basis of a vector space is a set of linearly independent vectors that span the full space."
### 9. What is Ax = b? When does Ax =b has a unique solution?
### 10. In Ax = b, what happens when A is fat or tall?
### 11. When does inverse of A exist?
### 12. What is a norm? What is L1, L2 and L infinity norm?
### 13. What are the conditions a norm has to satisfy?
### 14. Why is squared of L2 norm preferred in ML than just L2 norm?
### 15. When L1 norm is preferred over L2 norm?
### 16. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?
### 17. What is Frobenius norm?
### 18. What is a diagonal matrix?
### 19. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
### 20. At what conditions does the inverse of a diagonal matrix exist?
### 21. What is a symmetrix matrix?
### 22. What is a unit vector?
### 23. When are two vectors x and y orthogonal?
### 24. At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?
### 25. When are two vectors x and y orthonormal?
### 26. What is an orthogonal matrix? Why is computationally preferred?
### 27. What is eigendecomposition, eigenvectors and eigenvalues?
### 28. How to find eigen values of a matrix?
### 29. Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
### 30. Is the Eigendecomposition guaranteed to be unique? If not, then how do we represent it?
### 31. What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
### 32. What is Singular Value Decomposition? Why do we use it? Why not just use ED?
### 33. Given a matrix A, how will you calculate its Singular Value Decomposition?
### 34. What are singular values, left singulars and right singulars?
### 35. What is the connection of Singular Value Decomposition of A with functions of A?
### 36. Why are singular values always non-negative?
### 37. What is the Moore Penrose pseudo inverse and how to calculate it?
### 38. If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
### 39. Which matrices can be decomposed by ED?
### 40. Which matrices can be decomposed by SVD?
### 41. What is the trace of a matrix?
### 42. How to write Frobenius norm of a matrix A in terms of trace?
### 43. Why is trace of a multiplication of matrices invariant to cyclic permutations?
### 44. What is the trace of a scalar?
### 45. Write the frobenius norm of a matrix in terms of trace?



