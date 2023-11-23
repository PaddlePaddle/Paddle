import pysnooper

import paddle
from paddle.distributed.communication.group import is_initialized

@pysnooper.snoop(output=f"snooper{paddle.distributed.get_rank()}.log", depth=1, max_variable_length=200)
def get_read_items(path, state_dict, process_group):
    print(f"pure hang test", flush=True)
    # for param_name, val in state_dict.items():
    for param_name, val in enumerate(range(2)):
        if True or isinstance(val, paddle.Tensor):
            print(f"before val:{val}, type:{type(val)}", flush=True)
            if True or val.is_dist():
                paddle.distributed.barrier()
                # pass
                # local_shape, global_offset = compute_local_shape_and_global_offset(val.shape, val.dist_attr.process_mesh, val.dist_attr.dims_mapping)
                # cur_chunk_metadata = ChunkMetadata(local_shape, global_offset)
                # assert param_name in param_to_chunkmetadata, f"param_name:{param_name} not found in param_to_chunkmetadata:{param_to_chunkmetadata}."
                # for storage_chunk_metadata in param_to_chunkmetadata[param_name]:
                for storage_chunk_metadata in range(2):
                    print(f"rank:{paddle.distributed.get_rank()}, storage_chunk_metadata:{storage_chunk_metadata}", flush=True)
                    # paddle.distributed.barrier()
                    print(f"param_name:{param_name}, storage_chunk_metadata:{storage_chunk_metadata}")
                    if paddle.distributed.get_rank() == 0 or paddle.distributed.get_rank() == 1:
                        continue
                    else:
                        continue
            else:
                print(f"val:{val}, type:{type(val)}")
                pass
        else:
            pass
    return

def main():
    path = "./output"
    ###!!! Init the Disttensor and turn on the pysnooper at the same time will lead to hang !!!

    # import paddle.distributed as dist
    # w1 = paddle.arange(8).reshape([4, 2])
    # w2 = paddle.arange(8, 12).reshape([2, 2])
    # mesh = dist.ProcessMesh([0,1,2,3], dim_names=["x"])
    # w1_dist_attr = dist.DistAttr(mesh, sharding_specs=["x", None])
    # sharded_w1 = dist.shard_tensor(w1, dist_attr=w1_dist_attr)
    # w2_dist_attr = dist.DistAttr(mesh, sharding_specs=[None, None])
    # sharded_w2 = dist.shard_tensor(w2, dist_attr=w2_dist_attr)
    # state_dict = {"w1": sharded_w1, "w2": sharded_w2}

    not is_initialized() and paddle.distributed.init_parallel_env()
    get_read_items(path, None, None)

if __name__ == "__main__":
    main()