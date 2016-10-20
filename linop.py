"""This file contains subclasses of scipy's LinearOperator to represent sparse linear operators with various structure."""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import _ScaledLinearOperator, _SumLinearOperator, _ProductLinearOperator
from scipy.sparse import csr_matrix, lil_matrix
from functools import reduce

class BaseSparseLinearOperator(LinearOperator):
    def get(self, i, j):
        """Gets element i,j from the matrix representation of the SparseLinearOperator."""
        vi, vj = np.zeros(self.shape[0], dtype = int), np.zeros(self.shape[1], dtype = int)
        vi[i] = 1
        vj[j] = 1
        return np.dot(vi, self * vj)
    def todense(self):
        """Returns as a dense matrix. Warning: do not use this if the dimensions are large."""
        result = np.zeros(self.shape, dtype = float)
        for j in range(self.shape[1]):
            v = np.zeros(self.shape[1], dtype = float)
            v[j] = 1.0
            result[:, j] = self._matvec(v)
        return result
    def to_column_stochastic(self):
        """Returns a column-normalized version of the operator."""
        Dinv = DiagonalLinearOperator(1.0 / (self.transpose() * np.ones(self.shape[0], dtype = float)))
        return (self * Dinv)
    def to_row_stochastic(self):
        """Returns a row-normalized version of the operator."""
        Dinv = DiagonalLinearOperator(1.0 / (self * np.ones(self.shape[1], dtype = float)))
        return (Dinv * self)
    def __add__(self, other):
        if isinstance(other, BaseSparseLinearOperator):
            return SumLinearOperator(self, other)
        return SumLinearOperator(self, other * RankOneLinearOperator(np.ones(self.shape[0]), np.ones(self.shape[1])))
    def __radd__(self, other):
        if isinstance(other, BaseSparseLinearOperator):
            return SumLinearOperator(other, self)
        return SumLinearOperator(other * RankOneLinearOperator(np.ones(self.shape[0]), np.ones(self.shape[1])), self)
    def __mul__(self, other):
        if isinstance(other, BaseSparseLinearOperator):
            return ProductLinearOperator(self, other)
        elif isinstance(other, np.ndarray):
            return self._matvec(other)
        return ScaledLinearOperator(self, other)
    def __rmul__(self, other):
        if isinstance(other, BaseSparseLinearOperator):
            return ProductLinearOperator(other, self)
        elif isinstance(other, np.ndarray):
            return self.transpose()._matvec(other)
        return ScaledLinearOperator(self, other)
    def __sub__(self, other):
        return self.__add__(-other)
    def __rsub__(self, other):
        return (-self).__radd__(other)
    def __neg__(self):
        return self * -1.0
    def __repr__(self):
        s = super().__repr__()
        if (self.shape[0] * self.shape[1] <= 10000):
            s += '\n' + str(self.todense())
        return s

class SparseLinearOperator(BaseSparseLinearOperator):
    """Subclass of LinearOperator for handling a sparse matrix."""
    def __init__(self, F):
        """Input must be a sparse matrix or SparseLinearOperator."""
        self.F = F
        super().__init__(dtype = float, shape = F.shape)
    def _matvec(self, x):
        return self.F * x
    def _transpose(self):
        return SparseLinearOperator(self.F.transpose())
    def __getnewargs__(self):  # for pickling
        return (self.F,)

class ScaledLinearOperator(_ScaledLinearOperator, BaseSparseLinearOperator):
    def _transpose(self):
        return ScaledLinearOperator(self.args[0].transpose(), self.args[1])
    def __getnewargs__(self):
        return self.args

class SumLinearOperator(_SumLinearOperator, BaseSparseLinearOperator):
    def _transpose(self):
        return SumLinearOperator(self.args[0].transpose(), self.args[1].transpose())
    def __getnewargs__(self):
        return self.args

class ProductLinearOperator(_ProductLinearOperator, BaseSparseLinearOperator):
    def _transpose(self):
        return ProductLinearOperator(self.args[1].transpose(), self.args[0].transpose())
    def __getnewargs__(self):
        return self.args

class SymmetricSparseLinearOperator(SparseLinearOperator):
    """Linear operator whose adjoint operator is the same, due to symmetry."""
    def _adjoint(self):
        return self
    def _transpose(self):
        return self

class DiagonalLinearOperator(SymmetricSparseLinearOperator):
    """Linear operator representing a diagonal matrix."""
    def __init__(self, D):
        """D is a 1D array containing the diagonal entries."""
        self.D = D
        LinearOperator.__init__(self, dtype = float, shape = (len(D), len(D)))
    def _matvec(self, x):
        return self.D * x
    def __getnewargs__(self):
        return (self.D,)

class ConstantDiagonalLinearOperator(DiagonalLinearOperator):
    """Linear operator representing a constant diagonal matrix."""
    def __init__(self, n, c):
        """n is dimension, c is a constant to be multiplied by the identity matrix."""
        self.c = c
        LinearOperator.__init__(self, dtype = float, shape = (n, n))
    def _matvec(self, x):
        return self.c * x
    def __getnewargs__(self):
        return (self.c,)

class RankOneLinearOperator(SparseLinearOperator):
    """Subclass of LinearOperator for handling a rank-one matrix. Represents the matrix u * v^T, where u and v are vectors."""
    def __init__(self, u, v):
        assert (len(u) == len(v))
        self.u, self.v = u, v
        LinearOperator.__init__(self, dtype = float, shape = (len(u), len(v)))
    def _transpose(self):
        return RankOneLinearOperator(self.v, self.u)
    def _matvec(self, x):
        return self.u * np.dot(self.v, x)
    def __getnewargs__(self):
        return (self.u, self.v)

class ReplicatedColumnLinearOperator(RankOneLinearOperator):
    """Subclass of RankOneLinearOperator for handling the case u * 1^T, i.e. the matrix whose columns are all u."""
    def __init__(self, u):
        self.u = u
        LinearOperator.__init__(self, dtype = float, shape = (len(u), len(u)))
    def _transpose(self):
        return ReplicatedRowLinearOperator(self.u)
    def _matvec(self, x):
        return self.u * np.sum(x)
    def __getnewargs__(self):
        return (self.u,)

class ReplicatedRowLinearOperator(RankOneLinearOperator):
    """Subclass of RankOneLinearOperator for handling the case 1 * u^T, i.e. the matrix whose rows are all u."""
    def __init__(self, u):
        self.u = u
        LinearOperator.__init__(self, dtype = float, shape = (len(u), len(u)))
    def _transpose(self):
        return ReplicatedColumnLinearOperator(self.u)
    def _matvec(self, x):
        return np.ones(len(self.u), dtype = float) * (np.dot(self.u, x))
    def __getnewargs__(self):
        return (self.u,)

class PMILinearOperator(SymmetricSparseLinearOperator):
    """Subclass of LinearOperator for handling the sparse + low-rank PMI matrix. In particular, it represents the matrix F + Delta * 1 * 1^T - u * 1^T - 1 * u^T."""
    def __init__(self, F, Delta, u):
        n = F.shape[0]
        assert ((F.shape[1] == n) and (u.shape == (n,)))
        self.F, self.Delta, self.u = F, Delta, u
        self.u_prime = self.Delta - self.u
        self.ones = np.ones(n, dtype = float)
        LinearOperator.__init__(self, dtype = float, shape = self.F.shape)
    def _matvec(self, x):
        return self.F * x + self.u_prime * np.sum(x) - self.ones * np.dot(self.u, x)
    def __getnewargs__(self):  # for pickling
        return (self.F, self.Delta, self.u)

class SparseLaplacian(SymmetricSparseLinearOperator):
    """Class for representing a sparse Laplacian (D - A). Can also subclass the normalized version (D^(-1/2) * A * D^(-1/2)) or the regularized normalized version ((D + tau * I)^(-1/2) * A * (D + tau * I)^(-1/2)). Constructs the Laplacian from a SymmetricSparseLinearOperator representing the adjacency matrix."""
    def __init__(self, A):
        assert isinstance(A, SymmetricSparseLinearOperator)
        self.A = A
        self.D = self.A.matvec(np.ones(self.A.shape[1], dtype = float))
        LinearOperator.__init__(self, dtype = float, shape = A.shape)
    def _matvec(self, x):
        return (self.D * x - self.A.matvec(x))
    def __getnewargs__(self):
        return (self.A,)

class SparseNormalizedLaplacian(SparseLaplacian):
    """Class representing the normalized Laplacian (D^(-1/2) * A * D^(-1/2))."""
    def __init__(self, A):
        super().__init__(A)
        self.D_inv_sqrt = 1.0 / np.sqrt(self.D)  # hopefully D has all positive entries
    def _matvec(self, x):
        return (x - self.D_inv_sqrt * self.A._matvec(self.D_inv_sqrt * x))

class SparseRegularizedNormalizedLaplacian(SparseLaplacian):
    """Class representing the regularized normalized Laplacian ((D + tau * I)^(-1/2) * A * (D + tau * I)^(-1/2))."""
    def __init__(self, A):
        super().__init__(A)
        tau = np.mean(self.D)
        self.D_plus_tau_I_inv_sqrt = 1.0 / np.sqrt(self.D + tau)
    def _matvec(self, x):
        return (x - self.D_plus_tau_I_inv_sqrt * self.A._matvec(self.D_plus_tau_I_inv_sqrt * x))

class SparseDiagonalAddedAdjacencyOperator(SymmetricSparseLinearOperator):
    """Class representing an adjacency matrix A + D/n."""
    def __init__(self, A):
        assert isinstance(A, SymmetricSparseLinearOperator)
        self.A = A
        LinearOperator.__init__(self, dtype = float, shape = A.shape)
        self.D_ratio = self.A._matvec(np.ones(self.A.shape[1], dtype = float)) / self.shape[0]
    def _matvec(self, x):
        return (self.A.matvec(x) + self.D_ratio * x)

class BlockSparseLinearOperator(SparseLinearOperator):
    """Class representing a block structure of sparse linear operators."""
    def __init__(self, block_grid):
        """Input is a 2D list of SparseLinearOperators. The resulting operator is the corresponding operator comprised of these operator blocks. The dimensions must match correctly. This assumes the number of blocks in each row and column is the same."""
        self.block_grid_shape = (len(block_grid), len(block_grid[0]))
        # validate block dimensions
        assert all([len(row) == self.block_grid_shape[1] for row in block_grid]), "Must be same number of blocks in each row."
        assert all([len(set([block_grid[i][j].shape[0] for j in range(self.block_grid_shape[1])])) == 1 for i in range(self.block_grid_shape[0])]), "dimension mismatch"
        assert all([len(set([block_grid[i][j].shape[1] for i in range(self.block_grid_shape[0])])) == 1 for j in range(self.block_grid_shape[1])]), "dimension mismatch"
        shape = (sum([block_grid[i][0].shape[0] for i in range(len(block_grid))]), sum([block_grid[0][j].shape[1] for j in range(len(block_grid[0]))]))
        # compute transition indices between blocks
        self.row_indices = [0] + list(np.cumsum([row[0].shape[0] for row in block_grid]))
        self.column_indices = [0] + list(np.cumsum([block.shape[1] for block in block_grid[0]]))
        self.block_grid = block_grid
        LinearOperator.__init__(self, dtype = float, shape = shape)
    def _matvec(self, x):
        result = np.zeros(self.shape[0], dtype = float)
        for i in range(self.block_grid_shape[0]):
            row = self.block_grid[i]
            partial_result = np.zeros(row[0].shape[0], dtype = float)
            for j in range(self.block_grid_shape[1]):
                partial_result += row[j]._matvec(x[self.column_indices[j] : self.column_indices[j + 1]])
            result[self.row_indices[i] : self.row_indices[i + 1]] = partial_result
        return result
    def __getnewargs__(self):  # for pickling
        return (self.block_grid,)

class CollapseOperator(SparseLinearOperator):
    """Given a mapping from [0 ... (n - 1)] to Pow([0 ... (m - 1)]), represents the m x n linear operator that sums together vector entries belonging to the same equivalence class under this mapping."""
    def __init__(self, mapping, m):
        """mapping is an n-long array of sets of integers in [0 ... (m - 1)]."""
        self.mapping = mapping
        self.m = m
        n = len(mapping)
        #assert (m <= n)
        assert (m > max((reduce(max, s, -1) for s in mapping)))
        assert (min(map(len, mapping)) > 0)
        mat = lil_matrix((m, n), dtype = float)
        for (i, img) in enumerate(mapping):
            for j in img:
                mat[j, i] = 1.0 / len(img)
        SparseLinearOperator.__init__(self, mat.tocsr())
    def __getnewargs__(self):  # for pickling
        return (self.mapping, self.m)

class JointSymmetricBlockOperator(SymmetricSparseLinearOperator):
    """Given a list of SymmetricSparseLinearOperators of the same dimension, constructs a joint operator where the diagonal blocks are the input operators, and the off-diagonal blocks are the means of these operators."""
    def __init__(self, diag_blocks, tau = None):
        self.diag_blocks = diag_blocks
        self.tau = tau
        self.num_blocks = len(diag_blocks)
        assert (isinstance(diag_blocks, list) and len(diag_blocks) > 0)
        assert all([isinstance(block, SymmetricSparseLinearOperator) for block in diag_blocks]), "Blocks must be symmetric."
        assert all([block.shape == diag_blocks[0].shape for block in diag_blocks]), "Blocks must have the same shape."
        self.n = diag_blocks[0].shape[0]
        if (tau is None):
            joint_block_operator = BlockSparseLinearOperator([[diag_blocks[i] if (i == j) else (0.5 * np.sum([diag_blocks[i], diag_blocks[j]])) for j in range(self.num_blocks)] for i in range(self.num_blocks)])
        else:
            scaled_identity = ConstantDiagonalLinearOperator(diag_blocks[0].shape[0], tau)
            joint_block_operator = BlockSparseLinearOperator([[diag_blocks[i] if (i == j) else (0.5 * np.sum([diag_blocks[i], diag_blocks[j]]) + scaled_identity) for j in range(self.num_blocks)] for i in range(self.num_blocks)])
        SymmetricSparseLinearOperator.__init__(self, joint_block_operator)
        def __getnewargs__(self):
            return (self.diag_blocks, self.tau)


class RowSelectorOperator(SparseLinearOperator):
    """Given a dimension n and a dictionary mapping rows to indices (for m rows), creates an operator representing the m x n matrix selecting the entries at the given indices. Rows with no indices listed provide an entry of 0. Alternatively if ind_map is a list of indices, there will be no zero rows."""
    def __init__(self, m, n, ind_map):
        self.m, self.n = m, n
        if isinstance(ind_map, (list, range, np.ndarray)):
            ind_map = dict(enumerate(ind_map))
        self.ind_map = ind_map
        s = set(ind_map.values())
        assert all([0 <= i < n for i in s])
        num_entries = len(ind_map)
        self.unique = (num_entries == len(s))
        self.row_map = np.zeros((num_entries, 2), dtype = int)
        for (ctr, item) in enumerate(sorted(ind_map.items(), key = lambda pair : pair[0])):
            self.row_map[ctr] = item
        LinearOperator.__init__(self, dtype = float, shape = (m, n))
    def _matvec(self, x):
        y = np.zeros(self.shape[0], dtype = x.dtype)
        y[self.row_map[:, 0]] = x[self.row_map[:, 1]]
        return y
    def _transpose(self):
        m = self.shape[0]
        if self.unique:
            return RowSelectorOperator(self.n, self.m, {i : row for (row, i) in self.row_map})
        else:
            F = lil_matrix((self.n, self.m), dtype = float)
            for (row, i) in self.row_map:
                F[i, row] = 1.0
            return SparseLinearOperator(F)
    def __getnewargs__(self):
        return (self.m, self.n, self.ind_map)

class PermutationOperator(RowSelectorOperator):
    def __init__(self, perm):
        """Given a permutation on n indices, constructs the n x n permutation matrix selecting rows in the permutation order."""
        self.n = len(perm)
        assert all([0 <= i < self.n for i in perm])
        assert (self.n == len(set(perm)))
        self.perm = np.asarray(perm)
        LinearOperator.__init__(self, dtype = float, shape = (self.n, self.n))
    def _matvec(self, x):
        return x[self.perm]
    def _transpose(self):
        inv_perm = np.zeros(self.n, dtype = int)
        for (i, j) in enumerate(self.perm):
            inv_perm[j] = i
        return PermutationOperator(inv_perm)
    def inv(self):
        return self.transpose()
    def __getnewargs__(self):
        return (self.perm,)


