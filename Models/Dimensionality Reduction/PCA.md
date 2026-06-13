Principal Component Analysis: Find a linear and orthogonal projection of the high dimensional data $x \in R^D$ to a lower dimensional subspace $z \in R^L$

This is a good approximation in the following sense:
- Project or encode $x$ to get $z = W^Tx$
- unproject or decode $z$ to get $\hat{x} = Wz$
- Goal is to have $\hat{x}$ to be as close to $x$ in $\mathcal{l}_2$ space

The reconstruction error is: $$
\mathcal{L}(\mathbf{W}) \triangleq \frac{1}{N} \sum_{n=1}^N\left\|\boldsymbol{x}_n-\operatorname{decode}\left(\operatorname{encode}\left(\boldsymbol{x}_n ; \mathbf{W}\right) ; \mathbf{W}\right)\right\|_2^2
$$
This function is minimized 