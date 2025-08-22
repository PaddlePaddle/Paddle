import paddle
import paddle.distributed as dist

# Initialize parallel environment
dist.init_parallel_env()
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create a tensor on GPU
x = paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32')
y = paddle.to_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype='float32')

# Perform AllReduce (across all GPUs)
print(f"Before AllReduce sum on rank {dist.ParallelEnv().rank}: {x.numpy()}")
handle = dist.stream.all_reduce(x, op=dist.ReduceOp.SUM, sync_op=False)
handle.wait()
print(f"After AllReduce sum on rank {dist.ParallelEnv().rank}: {x.numpy()}")

print(f"Before AllReduce min on rank {dist.ParallelEnv().rank}: {y.numpy()}")
dist.stream.all_reduce(x, op=dist.ReduceOp.MIN, sync_op=True)
print(f"After AllReduce min on rank {dist.ParallelEnv().rank}: {y.numpy()}")

# Perform AllGather (across all GPUs)
tensor_list = []
print(f"Before AllGather on rank {dist.ParallelEnv().rank}: x = {x.numpy()}, tensor_list = {tensor_list}")
dist.all_gather(tensor_list, x)
print(f"After AllGather on rank {dist.ParallelEnv().rank}: x = {x.numpy()}, tensor_list = {tensor_list}")

# Perform AlltoAll (across all GPUs)
in_tensor_list = []
for i in range(world_size):
    in_tensor_list.append(paddle.to_tensor([rank * 10 + i], dtype='float32'))
out_tensor_list = []
print(f"Before AlltoAll on rank {dist.ParallelEnv().rank}: in_tensor_list = {in_tensor_list}, out_tensor_list = {out_tensor_list}")
handle = dist.stream.alltoall(out_tensor_list, in_tensor_list, sync_op=False)
handle.wait()
print(f"After AlltoAll on rank {dist.ParallelEnv().rank}: in_tensor_list = {in_tensor_list}, out_tensor_list = {out_tensor_list}")

# Perform Broadcast from GPU 1
x = paddle.to_tensor([rank + 2], dtype='float32')
print(f"Before Broadcast (from rank 0) on rank {dist.ParallelEnv().rank}: x = {x}")
dist.broadcast(x, src=1)
print(f"After Broadcast (from rank 0) on rank {dist.ParallelEnv().rank}: x = {x}")
