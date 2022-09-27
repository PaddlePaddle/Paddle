#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import copy
import paddle

# Use to store the previous and current process mesh
_g_previous_process_mesh = None
_g_current_process_mesh = None


def get_current_process_mesh():
    global _g_current_process_mesh
    return _g_current_process_mesh


def set_current_process_mesh(process_mesh):
    global _g_previous_process_mesh
    global _g_current_process_mesh
    _g_previous_process_mesh = _g_current_process_mesh
    _g_current_process_mesh = process_mesh


def reset_current_process_mesh():
    global _g_previous_process_mesh
    global _g_current_process_mesh
    _g_current_process_mesh = _g_previous_process_mesh


class ProcessMesh(object):
    """
    The `Processmesh` object describes the topology of the used processes.

    Args:
        mesh (list|numpy.array): an n-dimensional array describes the toplogy
            of the processes.
        dim_names (list, optional): the i-th element of this list gives the name of the
            i-th dimension of the mesh.

    Examples:
        .. code-block:: python

            import paddle

            mesh = auto.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
            assert mesh.shape == [2, 3]
            assert mesh.processe_ids == [2, 4, 5, 0, 1, 3]

    """

    def __init__(self, mesh=None, dim_names=None, shape=None, process_ids=None):
        # Use shape and process_ids just for compatibility
        # Users should not use these directly
        if mesh is None:
            assert shape is not None
            assert process_ids is not None
            mesh = np.array(process_ids).reshape(shape)

        if not isinstance(mesh, list) and \
           not isinstance(mesh, np.ndarray):
            raise ValueError(
                'The mesh must be an instance of list or np.ndarray.')
        if isinstance(mesh, list):
            mesh = np.array(mesh)

        self._mesh = mesh
        self._shape = list(self._mesh.shape)
        self._process_ids = self._mesh.flatten().tolist()

        assert all(isinstance(p, int) for p in self._process_ids), \
            ("All elements of the mesh must be integer")
        assert min(
            self._process_ids) >= 0, ('All elements of the mesh must be >= 0.')
        unique_process_ids = set(self._process_ids)
        assert len(unique_process_ids) == len(
            self._process_ids), ('All elements of the mesh must be unique.')

        if dim_names is not None:
            assert len(dim_names) == len(self._shape), \
                ("The length of dims_names must be same as the shape of the mesh.")
            self._dim_names = copy.deepcopy(dim_names)
        else:
            self._dim_names = ["d" + str(i) for i in range(len(self._shape))]
        unique_dim_names = set(self._dim_names)
        assert len(unique_dim_names) == len(self._dim_names), (
            'All dim_names {} must be unique.'.format(dim_names))

        # Store all process meshes
        from .dist_context import get_default_distributed_context
        default_dist_cxt = get_default_distributed_context()
        default_dist_cxt.add_process_mesh(self)
        # Add new processes to process group 0
        from .process_group import get_process_group
        pg0 = get_process_group(0)
        pg0.add_ranks(self.processes)

    @property
    def shape(self):
        """
        Get the shape of this ProcessMesh.
        """
        return self._shape

    @property
    def process_ids(self):
        """
        Get the process ids belonging to this ProcessMesh.
        """
        return self._process_ids

    @property
    def dim_names(self):
        """
        Get the dimension names of this ProcessMesh.
        """
        return self._dim_names

    @property
    def ndim(self):
        """
        Get the number of dimension of this ProcessMesh.
        """
        return len(self._shape)

    @property
    def mesh(self):
        """
        Get the underlying mesh of ProcessMesh.
        """
        return self._mesh

    @property
    def topology(self):
        return self._shape

    @property
    def processes(self):
        return self._process_ids

    def __getitem__(self, index):
        if isinstance(index, tuple):
            new_dim_names = []
            for i, item in enumerate(index):
                if isinstance(item, slice):
                    new_dim_names.append(self._dim_names[i])
            new_mesh = self._mesh[index]
            if new_mesh.shape:
                return ProcessMesh(new_mesh, new_dim_names)
            else:
                # Wrap a scalar into a list but without dim_names
                return ProcessMesh([new_mesh])
        elif isinstance(index, slice):
            new_mesh = self._mesh[index]
            new_dim_names = self._dim_names
            return ProcessMesh(new_mesh, new_dim_names)
        else:
            new_mesh = self._mesh[index]
            new_dim_names = self._dim_names[1:]
            return ProcessMesh(new_mesh, new_dim_names)

    def __enter__(self):
        set_current_process_mesh(self)
        default_prog = paddle.fluid.default_main_program()
        cur_block = default_prog.current_block()
        self._old_var_names = list(cur_block.vars.keys())
        self._old_op_size = len(cur_block.ops)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        from .dist_tensor import DistributedTensor
        from .dist_op import DistributedOperator
        default_prog = paddle.fluid.default_main_program()
        cur_block = default_prog.current_block()
        new_var_names = list(cur_block.vars.keys())
        new_op_size = len(cur_block.ops)
        from .dist_context import get_default_distributed_context
        default_dist_ctx = get_default_distributed_context()
        for name in new_var_names:
            if name not in self._old_var_names:
                tensor = cur_block.vars[name]
                dist_tensor = default_dist_ctx.get_dist_tensor_for_program(
                    tensor)
                if dist_tensor is None:
                    dist_tensor = DistributedTensor(cur_block.vars[name],
                                                    {"process_mesh": self})
                    dist_tensor.dist_attr.mark_annotated("process_mesh")
                    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
                else:
                    if dist_tensor.dist_attr.process_mesh is None:
                        dist_tensor.dist_attr.process_mesh = self
                        dist_tensor.dist_attr.mark_annotated("process_mesh")

        for idx in range(self._old_op_size, new_op_size):
            op = cur_block.ops[idx]
            dist_op = default_dist_ctx.get_dist_op_for_program(op)
            if dist_op is None:
                dist_op = DistributedOperator(op, {"process_mesh": self})
                dist_op.dist_attr.mark_annotated("process_mesh")
                default_dist_ctx.add_dist_op_for_program(dist_op)
            else:
                if dist_op.dist_attr.process_mesh is None:
                    dist_op.dist_attr.process_mesh = self
                    dist_op.dist_attr.mark_annotated("process_mesh")
        reset_current_process_mesh()

    def __eq__(self, other):
        if not isinstance(other, ProcessMesh):
            return False
        if self.shape != other.shape or self.process_ids != other.process_ids:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        str = "shape {}, process_ids {}, dim_nams {}".format(
            self.shape, self.process_ids, self.dim_names)
        return str
