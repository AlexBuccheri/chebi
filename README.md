# Algorithms

### Orthogonalisation

* Gram-Schmidt
  * Classic version needs more testing, and documenting
  * Modified version appears to work but should be documented

* Inner product approximation to go with Gram-Schmit 
  * Work through [this](https://sites.cs.ucsb.edu/~gilbert/cs290hSpr2014/Projects/BiegertKonoplivProject.pdf) 
    reference

### Iterative Eigensolvers

* Lanczos
  * References:
    * https://github.com/zachtheyek/Lanczos-Algorithm/blob/master/src/lanczos.ipynb
    * https://sites.cs.ucsb.edu/~gilbert/cs290hSpr2014/Projects/BiegertKonoplivProject.pdf
    * https://en.wikipedia.org/wiki/Lanczos_algorithm

* Chebyshev filtering
  * Requires: Lanczos, Gram-Schmidt, matrix-matrix, multiplication
  * Should prototype by calling library routines where possible

* [Rayleigh_quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient_iteration)
