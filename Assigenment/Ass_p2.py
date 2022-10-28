from numba import cuda
import numba
import numpy as np
from scipy.sparse import coo_matrix
import math

param = 1000
N=100

t_per_block = 20  # <=32
block_per_grid = int(np.ceil((N + 1) / t_per_block))
nelements = ((N - 1) ** 2 + N - 1) * 4 + (N + 1) * 3 - 2


@cuda.jit
def generate_tem_mat(N, row_col, data, f):
    # row, column, data
    local_row_col = cuda.shared.array((2, t_per_block ** 2), numba.int64)
    local_data = cuda.shared.array((1, t_per_block ** 2), numba.float32)
    local_count = cuda.shared.array((1), numba.uint16)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    px, py = cuda.grid(2)
    if px > N or py > N: return

    cuda.syncthreads()

    location = numba.int32(py * (N + 1) + px)
    if px == 0 or px == N or py == 0:
        local_row_col[:, local_count] = [location, location]
        local_data[location] = 1.

        f[location] = 10 if i == 0 or i == N else 0
        local_count += 1

    else:
        local_row_col[:, local_count:local_count + 4] = [
            [location, location, location, location],
            [location - (N + 2), location - (N + 1), location - N, location]]
        local_data[:, local_count:local_count + 4] = [-N / param,
                                                      -1 + N / (param / 2),
                                                      -N / param,
                                                      1.]

        f[location] = 0
        local_count += 4

    cuda.syncthreads()

    local_row_col = local_row_col[:local_count]
    row_col = np.hstack([row_col, local_row_col])

    local_data = local_data[:local_count]
    data = np.hstack([data, local_data])


row_col = np.zeros((2, nelements), dtype=np.int64)
data = np.zeros((1, nelements), dtype=np.float32)

f = np.zeros((N + 1) ** 2, dtype=np.float32)

generate_tem_mat[(block_per_grid, block_per_grid), (t_per_block, t_per_block)](
    N, row_col, data, f)

A = coo_matrix((result[2, :], (result[0, :], result[1, :])),
               shape=((N + 1) ** 2, (N + 1) ** 2)).tocsr()
#%%
