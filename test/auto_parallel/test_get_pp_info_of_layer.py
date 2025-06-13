# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddle.distributed import ProcessMesh
from paddle.distributed.auto_parallel.pipelining.utils import (
    GET_PP_INFO_OF_LAYER,
)


class TestGetPPInfoOfLayer(unittest.TestCase):
    def test_error_cases(self):
        # Test mesh dimension name error
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["dp"])
        with self.assertRaises(ValueError):
            GET_PP_INFO_OF_LAYER(
                hidden_layer_num=8, mesh=mesh, pp_schedule="1F1B"
            )

        # Test VPP without vpp_degree
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["pp"])
        with self.assertRaises(ValueError):
            GET_PP_INFO_OF_LAYER(
                hidden_layer_num=8, mesh=mesh, pp_schedule="VPP"
            )

        # Test insufficient layers for VPP
        with self.assertRaises(ValueError):
            GET_PP_INFO_OF_LAYER(
                hidden_layer_num=7, mesh=mesh, pp_schedule="VPP", vpp_degree=2
            )

        # Test unsupported schedule
        with self.assertRaises(ValueError):
            GET_PP_INFO_OF_LAYER(
                hidden_layer_num=8,
                mesh=mesh,
                pp_schedule="VPP_ZERO",
                vpp_degree=2,
            )

        # Test layer index out of range
        with self.assertRaises(ValueError):
            pp_info = GET_PP_INFO_OF_LAYER(
                hidden_layer_num=8, mesh=mesh, pp_schedule="1F1B", vpp_degree=2
            )
            pp_info[9]

    def test_1F1B_and_GPipe(self):
        # Test 1D mesh
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["pp"])
        pp_info_1F1B = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="1f1B"
        )
        pp_info_Gpipe = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="Gpipe"
        )

        self.assertEqual(pp_info_1F1B[3].process_ids, [1])
        self.assertEqual(
            pp_info_Gpipe[3].process_ids, pp_info_1F1B[3].process_ids
        )

        # Test 2D mesh
        mesh = ProcessMesh([[0, 1], [2, 3]], dim_names=["pp", "dp"])
        pp_info_1F1B = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="1F1B"
        )
        pp_info_Gpipe = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="Gpipe"
        )

        self.assertEqual(pp_info_1F1B[3].process_ids, [0, 1])
        self.assertEqual(
            pp_info_Gpipe[3].process_ids, pp_info_1F1B[3].process_ids
        )

        # Test 3D mesh
        mesh = ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "mp"]
        )
        pp_info_1F1B = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="1F1B"
        )
        pp_info_Gpipe = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="Gpipe"
        )

        self.assertEqual(pp_info_1F1B[3].process_ids, [0, 1, 2, 3])
        self.assertEqual(
            pp_info_Gpipe[3].process_ids, pp_info_1F1B[3].process_ids
        )

    def test_VPP(self):
        # Test 1D mesh
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["pp"])
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="VPP", vpp_degree=2
        )
        self.assertEqual(pp_info[3].process_ids, [3])

        # Test 2D mesh
        mesh = ProcessMesh([[0, 1], [2, 3]], dim_names=["pp", "dp"])
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="VPP", vpp_degree=2
        )
        self.assertEqual(pp_info[2].process_ids, [2, 3])

        # Test 3D mesh
        mesh = ProcessMesh(
            [[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["pp", "dp", "mp"]
        )
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="VPP", vpp_degree=2
        )
        self.assertEqual(pp_info[3].process_ids, [4, 5, 6, 7])

    def test_uneven_distribution(self):
        # Test 1F1B with uneven layers
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["pp"])
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=7, mesh=mesh, pp_schedule="1F1B"
        )
        self.assertEqual(pp_info[5].process_ids, [2])
        self.assertEqual(pp_info[6].process_ids, [3])

        # Test VPP with uneven layers
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=9, mesh=mesh, pp_schedule="VPP", vpp_degree=2
        )
        self.assertEqual(pp_info[8].process_ids, [3])

    def test_info_mapping(self):
        mesh = ProcessMesh([0, 1, 2, 3], dim_names=["pp"])

        # Test 1F1B mapping
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="1F1B"
        )
        layer_mesh_of_pp = pp_info.get_info_mapping()
        for layer_idx, submesh in layer_mesh_of_pp.items():
            self.assertEqual(layer_idx // 2, submesh.process_ids[0])

        # Test VPP mapping
        pp_info = GET_PP_INFO_OF_LAYER(
            hidden_layer_num=8, mesh=mesh, pp_schedule="VPP", vpp_degree=2
        )
        layer_mesh_of_pp = pp_info.get_info_mapping()
        for layer_idx, submesh in layer_mesh_of_pp.items():
            self.assertEqual(layer_idx % 4, submesh.process_ids[0])


if __name__ == '__main__':
    unittest.main()
