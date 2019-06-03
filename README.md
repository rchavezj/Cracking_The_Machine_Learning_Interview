# Cracking The Machine Learning Interview
Solutions from the Medium article "Cracking the Machine Learning Interview" written by Subhrajit Roy
https://medium.com/subhrajit-roy/cracking-the-machine-learning-interview-1d8c5bb752d8

![alt_text](https://github.com/rchavezj/Cracking_The_Machine_Learning_Interview/blob/master/crackingTheMachineLearningInterviewCover.png)

# Contents:
|                        |                                          |
| ---------------------- | ---------------------------------------- |
| 1. [Linear Algebra](#Linear-Algebra)                         | 2. [Numerical Optimization](#Numerical-Optimization)                                         |
| 3. [Basics of Probability and Information Theory](#Basics-of-Probability-and-Information-Theory)                                                                                                        |  4. [Confidence Interval](#Confidence-Interval)|
| 5. [Learning Theory](#Learning-Theory)                       |  6. [Model and Feature Selection](#Model-and-Feature-Selection) |
| 7. [Curse of dimensionality](#Curse-of-Dimensionality)       |  8. [Universal approximation of neural networks](#Universal-Approximation-of-Neural-Networks) |
| 9. [Deep Learning motivation](#Deep-Learning-Motivation)     |  10. [Support Vector Machine](#Support-Vector-Machine) |
| 11. [Bayesian Machine Learning](#Bayesian-Machine-Learning)  |  12. [Regularization](#Regularization) |
| 13. [Evaluation of Machine Learning systems](#Evaluation-of-Machine-Learning-Systems) |  14. [Clustering](#Clustering)  |
| 15. [Dimensionality Reduction](#Dimensionality-Reduction)    |  16. [Basics of Natural Language Processing](#Basics-of-Natural-Language-Processing) |
| 17. [Some basic questions](#Some-basic-questions)            |  18. [Optimization Procedures](#Optimization-Procedures) |
| 19. [Sequence Modeling](#Sequence-Modeling)                  |  20. [Autoencoders](#Autoencoders)               |
| 21. [Representation Learning](#Representation-Learning)      |  22. [Monte Carlo Methods](#Monte-Carlo-Methods) |


### Linear Algebra

1. What is broadcasting in connection to Linear Algebra?
2. What are scalars, vectors, matrices, and tensors?
3. What is Hadamard product of two matrices?
4. What is an inverse matrix?
5. If inverse of a matrix exists, how to calculate it?
6. What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?
7. Discuss span and linear dependence.
8. What is Ax = b? When does Ax =b has a unique solution?
9. In Ax = b, what happens when A is fat or tall?
10. When does inverse of A exist?
11. What is a norm? What is L1, L2 and L infinity norm?
12. What are the conditions a norm has to satisfy?
13. Why is squared of L2 norm preferred in ML than just L2 norm?
14. When L1 norm is preferred over L2 norm?
15. Can the number of nonzero elements in a vector be defined as L0 norm? If no, why?
16. What is Frobenius norm?
17. What is a diagonal matrix?
18. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
19. At what conditions does the inverse of a diagonal matrix exist?
20. What is a symmetrix matrix?
21. What is a unit vector?
22. When are two vectors x and y orthogonal?
At R^n what is the maximum possible number of orthogonal vectors with non-zero norm?
When are two vectors x and y orthonormal?
What is an orthogonal matrix? Why is computationally preferred?
What is eigendecomposition, eigenvectors and eigenvalues?
How to find eigen values of a matrix?
Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
Is the Eigendecomposition guaranteed to be unique? If not, then how do we represent it?
What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
What is Singular Value Decomposition? Why do we use it? Why not just use ED?
Given a matrix A, how will you calculate its Singular Value Decomposition?
What are singular values, left singulars and right singulars?
What is the connection of Singular Value Decomposition of A with functions of A?
Why are singular values always non-negative?
What is the Moore Penrose pseudo inverse and how to calculate it?
If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
Which matrices can be decomposed by ED?
Which matrices can be decomposed by SVD?
What is the trace of a matrix?
How to write Frobenius norm of a matrix A in terms of trace?
Why is trace of a multiplication of matrices invariant to cyclic permutations?
What is the trace of a scalar?
Write the frobenius norm of a matrix in terms of trace?
