from numba import cuda
import numba
import numpy as np
from scipy.sparse import coo_matrix
import math

param = 1000
N = 100
t_per_block = 20  # <=32
block_per_grid = int(math.ceil((N + 1) / t_per_block))
nelements = ((N - 1) ** 2 + N - 1) * 4 + (N + 1) * 3 - 2

total_t_per_block = t_per_block ** 2


@cuda.jit
def generate_tem_mat_cuda(N, rows, cols, data, f):
    total = total_t_per_block

    local_row = cuda.shared.array(shape=total, dtype=numba.int64)
    local_col = cuda.shared.array(shape=total, dtype=numba.int64)
    local_data = cuda.shared.array(shape=total, dtype=numba.float32)

    local_count = cuda.shared.array(shape=1, dtype=numba.int64)

    global_count = 0

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    px, py = cuda.grid(2)

    cuda.syncthreads()
    if px > N or py > N:
        pass
    else:

        location = py * (N + 1) + px

        if px == 0 or px == N or py == 0:
            local_row[local_count[0]] = location
            local_col[local_count[0]] = location
            local_data[location] = 1.

            # f[location] = 10 if px == 0 or px == N else 0
            local_count[0] += 1

        else:
            for i in range(4):
                local_row[local_count[0] + i] = location

                local_row[local_count[0] + i] = location if i == 3 else location - (N + 2) + i

            local_data[local_count[0]] = local_data[local_count[0] + 2] = -N / param
            local_data[local_count[0] + 1] = -1 + N / (param / 2)
            local_data[local_count[0] + 3] = 1.

            # f[location] = 0
            local_count[0] += 4

    cuda.syncthreads()

    # for i in range(local_count[0]):

    #     rows[global_count + i] = local_row[i]
    #     cols[global_count + i] = local_col[i]
    #     data[global_count + i] = local_data[i]

    # global_count += local_count[0]
    # print(global_count)


rows = np.zeros(shape=nelements, dtype=np.int64)
cols = np.zeros(nelements, dtype=np.int64)

data = np.zeros(nelements, dtype=np.float32)
f = np.zeros((N + 1) ** 2, dtype=np.float32)

blocks = (block_per_grid, block_per_grid)
threads = (t_per_block, t_per_block)
print('blocks/grid: {}\nthreads/block: {}'.format(block_per_grid, t_per_block))

generate_tem_mat_cuda[(block_per_grid, block_per_grid), (t_per_block, t_per_block)](N, rows, cols, data, f)

A = coo_matrix((data, (rows, cols)),
               shape=((N + 1) ** 2, (N + 1) ** 2)).tocsr()