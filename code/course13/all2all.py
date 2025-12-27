# alltoall_equal.pyfrom mpi4py import MPI
import numpy as np
# alltoall_equal.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# 准备发送缓冲区：一个二维数组 shape=(size, k)
k = 2
sendbuf = np.arange(rank*size*k, (rank+1)*size*k, dtype='i').reshape(size, k)
# 发送给自己的那一行可用作“自环”数据
print(f"Rank {rank} sendbuf=\n{sendbuf}")

# 预先分配接收同形数组
recvbuf = np.empty_like(sendbuf)
comm.Alltoall(sendbuf, recvbuf)

print(f"Rank {rank} recvbuf=\n{recvbuf}")
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# 准备发送缓冲区：一个二维数组 shape=(size, k)
k = 2
sendbuf = np.arange(rank*size*k, (rank+1)*size*k, dtype='i').reshape(size, k)
# 发送给自己的那一行可用作“自环”数据
print(f"Rank {rank} sendbuf=\n{sendbuf}")

# 预先分配接收同形数组
recvbuf = np.empty_like(sendbuf)
comm.Alltoall(sendbuf, recvbuf)
print(f"Rank {rank} recvbuf=\n{recvbuf}")
