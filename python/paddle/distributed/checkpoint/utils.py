#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import copy
from typing import List, Optional

import numpy as np
import paddle
from paddle.framework import core

def get_coordinator(mesh:np.array, rank:int):
    mesh = paddle.to_tensor(mesh)
    rand_coordinator = (mesh == rank).nonzero()
    assert rand_coordinator.shape[0] in (0, 1), f"rand_coordinator.shape: {rand_coordinator.shape}"
    return rand_coordinator[0].tolist() if rand_coordinator.shape[0] > 0 else None

# TODO(pangengzheng): support DeviceMesh and Placement later, device_mesh:Optional[core.ProcessMesh, core.DeviceMesh], placements:Optional[List[int], core.Placement]
def compute_local_shape_and_global_offset(global_shape:List[int], process_mesh:core.ProcessMesh, dims_mapping:List[int]) -> Tuple[Tuple[int], Tuple[int]]:
    """
    tensor dist_attr look like: {process_mesh: {shape: [2], process_ids: [0,1], dim_names: [x]}, dims_mapping: [-1,0], batch_dim: 0, dynamic_dims: [], annotated: [dims_mapping: 1,process_mesh: 1], partial: [].}
    the tensor dims=2, dims_mapping means the dim0 is replicate, dim1 is shard by dim0 of process_mesh
    """
    mesh = np.array(process_mesh.process_ids).reshape(process_mesh.shape)
    # deal with cross mesh case
    if paddle.distributed.get_rank() not in mesh:
        return ((), ())
    rank_coordinator = get_coordinator(mesh, paddle.distributed.get_rank())
    local_shape = copy.copy(global_shape)
    global_offset = [0 for _ in global_shape]
    # print(f"rank_coordinator:{rank_coordinator}")
    for i, dim in enumerate(dims_mapping):
        if dim == -1:
            continue
        else:
            assert global_shape[i] % process_mesh.shape[dim] == 0, f"i:{i}, global_shape[i]:{global_shape[i]}, process_mesh.shape[dim]:{process_mesh.shape[dim]}"
            local_shape[i] = global_shape[i] // process_mesh.shape[dim]
            chunk_idx = rank_coordinator[dim]
            global_offset[i] = chunk_idx * local_shape[i]
    
    return tuple(local_shape), tuple(global_offset)

def main_test():
    import paddle.distributed as dist

    tensor = paddle.arange(8).reshape([4, 2])
    global_shape = tensor.shape
    mesh = dist.ProcessMesh([[0,1], [2,3]], dim_names=["x", "y"])
    dist_attr = dist.DistAttr(mesh, sharding_specs=["x", "y"])
    sharded_tensor = dist.shard_tensor(tensor, dist_attr=dist_attr)
    print(f"get_tensor:{sharded_tensor.get_tensor().get_tensor()}, sharded_tensor.dist_attr:{sharded_tensor.dist_attr}")
    local_shape, global_offset = compute_local_shape_and_global_offset(global_shape, sharded_tensor.dist_attr.process_mesh, sharded_tensor.dist_attr.dims_mapping)
    print(f"local_shape:{local_shape}, global_offset: {global_offset}")

if __name__ == "__main__":
    main_test()
