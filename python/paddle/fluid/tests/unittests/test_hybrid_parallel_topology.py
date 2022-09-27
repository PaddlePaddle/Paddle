# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import unittest
from paddle.distributed import fleet
import numpy as np


class TestCommunicateTopology(unittest.TestCase):

    def test_topology(self):
        topo = fleet.CommunicateTopology(["dp", "mp", "pp"], [2, 2, 2])

        # test get_comm_list
        dp_comm_list = [[0, 4], [1, 5], [2, 6], [3, 7]]
        mp_comm_list = [[0, 2], [1, 3], [4, 6], [5, 7]]
        pp_comm_list = [[0, 1], [2, 3], [4, 5], [6, 7]]

        np.testing.assert_array_equal(dp_comm_list, topo.get_comm_list("dp"))
        np.testing.assert_array_equal(mp_comm_list, topo.get_comm_list("mp"))
        np.testing.assert_array_equal(pp_comm_list, topo.get_comm_list("pp"))

        # test get_hybrid_group_names
        parallel_names = ["dp", "mp", "pp"]
        np.testing.assert_array_equal(parallel_names,
                                      topo.get_hybrid_group_names())

        # test get_dims
        np.testing.assert_array_equal(2, topo.get_dim("dp"))
        np.testing.assert_array_equal(2, topo.get_dim("mp"))
        np.testing.assert_array_equal(2, topo.get_dim("pp"))

        # test world size
        self.assertEqual(topo.world_size(), 8)

        # test get_rank
        self.assertEqual(topo.get_rank(dp=0, mp=0, pp=0), 0)
        self.assertEqual(topo.get_rank(dp=0, mp=0, pp=1), 1)
        self.assertEqual(topo.get_rank(dp=0, mp=1, pp=0), 2)
        self.assertEqual(topo.get_rank(dp=0, mp=1, pp=1), 3)
        self.assertEqual(topo.get_rank(dp=1, mp=0, pp=0), 4)
        self.assertEqual(topo.get_rank(dp=1, mp=0, pp=1), 5)
        self.assertEqual(topo.get_rank(dp=1, mp=1, pp=0), 6)
        self.assertEqual(topo.get_rank(dp=1, mp=1, pp=1), 7)

        # test get_coord
        self.assertEqual(topo.get_coord(0), topo.coordinate(0, 0, 0))
        self.assertEqual(topo.get_coord(1), topo.coordinate(0, 0, 1))
        self.assertEqual(topo.get_coord(2), topo.coordinate(0, 1, 0))
        self.assertEqual(topo.get_coord(3), topo.coordinate(0, 1, 1))
        self.assertEqual(topo.get_coord(4), topo.coordinate(1, 0, 0))
        self.assertEqual(topo.get_coord(5), topo.coordinate(1, 0, 1))
        self.assertEqual(topo.get_coord(6), topo.coordinate(1, 1, 0))
        self.assertEqual(topo.get_coord(7), topo.coordinate(1, 1, 1))

        # test get_axis_list
        self.assertEqual(topo.get_axis_list("dp", 0), [0, 1, 2, 3])
        self.assertEqual(topo.get_axis_list("dp", 1), [4, 5, 6, 7])
        self.assertEqual(topo.get_axis_list("mp", 0), [0, 1, 4, 5])
        self.assertEqual(topo.get_axis_list("mp", 1), [2, 3, 6, 7])
        self.assertEqual(topo.get_axis_list("pp", 0), [0, 2, 4, 6])
        self.assertEqual(topo.get_axis_list("pp", 1), [1, 3, 5, 7])

        # test get_dim_size
        self.assertEqual(topo.get_dim_size("dp"), 2)
        self.assertEqual(topo.get_dim_size("mp"), 2)
        self.assertEqual(topo.get_dim_size("pp"), 2)

    def test_topology_4D(self):
        topo = fleet.CommunicateTopology(["dp", "pp", "sharding", "mp"],
                                         [2, 2, 2, 2])

        # test get_comm_list
        dp_comm_list = [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13],
                        [6, 14], [7, 15]]
        mp_comm_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
                        [12, 13], [14, 15]]
        pp_comm_list = [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13],
                        [10, 14], [11, 15]]
        sharding_comm_list = [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11],
                              [12, 14], [13, 15]]

        np.testing.assert_array_equal(dp_comm_list, topo.get_comm_list("dp"))
        np.testing.assert_array_equal(mp_comm_list, topo.get_comm_list("mp"))
        np.testing.assert_array_equal(pp_comm_list, topo.get_comm_list("pp"))
        np.testing.assert_array_equal(sharding_comm_list,
                                      topo.get_comm_list("sharding"))

        # test get_hybrid_group_names
        parallel_names = ["dp", "pp", "sharding", "mp"]
        np.testing.assert_array_equal(parallel_names,
                                      topo.get_hybrid_group_names())

        # test get_dims
        np.testing.assert_array_equal(2, topo.get_dim("dp"))
        np.testing.assert_array_equal(2, topo.get_dim("mp"))
        np.testing.assert_array_equal(2, topo.get_dim("pp"))
        np.testing.assert_array_equal(2, topo.get_dim("sharding"))

        # test world size
        self.assertEqual(topo.world_size(), 16)

        # test get_rank
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, mp=0), 0)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, mp=1), 1)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, mp=0), 2)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, mp=1), 3)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, mp=0), 4)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, mp=1), 5)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, mp=0), 6)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, mp=1), 7)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, mp=0), 8)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, mp=1), 9)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, mp=0), 10)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, mp=1), 11)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, mp=0), 12)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, mp=1), 13)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, mp=0), 14)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, mp=1), 15)

        # test get_coord
        self.assertEqual(topo.get_coord(0), topo.coordinate(0, 0, 0, 0))
        self.assertEqual(topo.get_coord(1), topo.coordinate(0, 0, 0, 1))
        self.assertEqual(topo.get_coord(2), topo.coordinate(0, 0, 1, 0))
        self.assertEqual(topo.get_coord(3), topo.coordinate(0, 0, 1, 1))
        self.assertEqual(topo.get_coord(4), topo.coordinate(0, 1, 0, 0))
        self.assertEqual(topo.get_coord(5), topo.coordinate(0, 1, 0, 1))
        self.assertEqual(topo.get_coord(6), topo.coordinate(0, 1, 1, 0))
        self.assertEqual(topo.get_coord(7), topo.coordinate(0, 1, 1, 1))
        self.assertEqual(topo.get_coord(8), topo.coordinate(1, 0, 0, 0))
        self.assertEqual(topo.get_coord(9), topo.coordinate(1, 0, 0, 1))
        self.assertEqual(topo.get_coord(10), topo.coordinate(1, 0, 1, 0))
        self.assertEqual(topo.get_coord(11), topo.coordinate(1, 0, 1, 1))
        self.assertEqual(topo.get_coord(12), topo.coordinate(1, 1, 0, 0))
        self.assertEqual(topo.get_coord(13), topo.coordinate(1, 1, 0, 1))
        self.assertEqual(topo.get_coord(14), topo.coordinate(1, 1, 1, 0))
        self.assertEqual(topo.get_coord(15), topo.coordinate(1, 1, 1, 1))

        # test get_axis_list
        self.assertEqual(topo.get_axis_list("dp", 0), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(topo.get_axis_list("dp", 1),
                         [8, 9, 10, 11, 12, 13, 14, 15])
        self.assertEqual(topo.get_axis_list("mp", 0),
                         [0, 2, 4, 6, 8, 10, 12, 14])
        self.assertEqual(topo.get_axis_list("mp", 1),
                         [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(topo.get_axis_list("pp", 0),
                         [0, 1, 2, 3, 8, 9, 10, 11])
        self.assertEqual(topo.get_axis_list("pp", 1),
                         [4, 5, 6, 7, 12, 13, 14, 15])
        self.assertEqual(topo.get_axis_list("sharding", 0),
                         [0, 1, 4, 5, 8, 9, 12, 13])
        self.assertEqual(topo.get_axis_list("sharding", 1),
                         [2, 3, 6, 7, 10, 11, 14, 15])

        # test get_dim_size
        self.assertEqual(topo.get_dim_size("dp"), 2)
        self.assertEqual(topo.get_dim_size("mp"), 2)
        self.assertEqual(topo.get_dim_size("pp"), 2)
        self.assertEqual(topo.get_dim_size("sharding"), 2)


if __name__ == '__main__':
    unittest.main()
