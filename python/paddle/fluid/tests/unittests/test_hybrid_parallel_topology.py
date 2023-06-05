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

import unittest

import numpy as np

from paddle.distributed import fleet


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
        np.testing.assert_array_equal(
            parallel_names, topo.get_hybrid_group_names()
        )

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
        topo = fleet.CommunicateTopology(
            ["dp", "pp", "sharding", "mp"], [2, 2, 2, 2]
        )

        # test get_comm_list
        dp_comm_list = [
            [0, 8],
            [1, 9],
            [2, 10],
            [3, 11],
            [4, 12],
            [5, 13],
            [6, 14],
            [7, 15],
        ]
        mp_comm_list = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
        ]
        pp_comm_list = [
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [8, 12],
            [9, 13],
            [10, 14],
            [11, 15],
        ]
        sharding_comm_list = [
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7],
            [8, 10],
            [9, 11],
            [12, 14],
            [13, 15],
        ]

        np.testing.assert_array_equal(dp_comm_list, topo.get_comm_list("dp"))
        np.testing.assert_array_equal(mp_comm_list, topo.get_comm_list("mp"))
        np.testing.assert_array_equal(pp_comm_list, topo.get_comm_list("pp"))
        np.testing.assert_array_equal(
            sharding_comm_list, topo.get_comm_list("sharding")
        )

        # test get_hybrid_group_names
        parallel_names = ["dp", "pp", "sharding", "mp"]
        np.testing.assert_array_equal(
            parallel_names, topo.get_hybrid_group_names()
        )

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
        self.assertEqual(
            topo.get_axis_list("dp", 1), [8, 9, 10, 11, 12, 13, 14, 15]
        )
        self.assertEqual(
            topo.get_axis_list("mp", 0), [0, 2, 4, 6, 8, 10, 12, 14]
        )
        self.assertEqual(
            topo.get_axis_list("mp", 1), [1, 3, 5, 7, 9, 11, 13, 15]
        )
        self.assertEqual(
            topo.get_axis_list("pp", 0), [0, 1, 2, 3, 8, 9, 10, 11]
        )
        self.assertEqual(
            topo.get_axis_list("pp", 1), [4, 5, 6, 7, 12, 13, 14, 15]
        )
        self.assertEqual(
            topo.get_axis_list("sharding", 0), [0, 1, 4, 5, 8, 9, 12, 13]
        )
        self.assertEqual(
            topo.get_axis_list("sharding", 1), [2, 3, 6, 7, 10, 11, 14, 15]
        )

        # test get_dim_size
        self.assertEqual(topo.get_dim_size("dp"), 2)
        self.assertEqual(topo.get_dim_size("mp"), 2)
        self.assertEqual(topo.get_dim_size("pp"), 2)
        self.assertEqual(topo.get_dim_size("sharding"), 2)

    def test_topology_5D(self):
        topo = fleet.CommunicateTopology(
            ["dp", "pp", "sharding", "sep", "mp"], [2, 2, 2, 2, 2]
        )

        # test get_comm_list
        dp_comm_list = [
            [0, 16],
            [1, 17],
            [2, 18],
            [3, 19],
            [4, 20],
            [5, 21],
            [6, 22],
            [7, 23],
            [8, 24],
            [9, 25],
            [10, 26],
            [11, 27],
            [12, 28],
            [13, 29],
            [14, 30],
            [15, 31],
        ]
        mp_comm_list = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23],
            [24, 25],
            [26, 27],
            [28, 29],
            [30, 31],
        ]
        pp_comm_list = [
            [0, 8],
            [1, 9],
            [2, 10],
            [3, 11],
            [4, 12],
            [5, 13],
            [6, 14],
            [7, 15],
            [16, 24],
            [17, 25],
            [18, 26],
            [19, 27],
            [20, 28],
            [21, 29],
            [22, 30],
            [23, 31],
        ]
        sharding_comm_list = [
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [8, 12],
            [9, 13],
            [10, 14],
            [11, 15],
            [16, 20],
            [17, 21],
            [18, 22],
            [19, 23],
            [24, 28],
            [25, 29],
            [26, 30],
            [27, 31],
        ]
        sep_comm_list = [
            [0, 2],
            [1, 3],
            [4, 6],
            [5, 7],
            [8, 10],
            [9, 11],
            [12, 14],
            [13, 15],
            [16, 18],
            [17, 19],
            [20, 22],
            [21, 23],
            [24, 26],
            [25, 27],
            [28, 30],
            [29, 31],
        ]

        np.testing.assert_array_equal(dp_comm_list, topo.get_comm_list("dp"))
        np.testing.assert_array_equal(mp_comm_list, topo.get_comm_list("mp"))
        np.testing.assert_array_equal(pp_comm_list, topo.get_comm_list("pp"))
        np.testing.assert_array_equal(
            sharding_comm_list, topo.get_comm_list("sharding")
        )
        np.testing.assert_array_equal(sep_comm_list, topo.get_comm_list("sep"))

        # test get_fused_ranks
        dp_sep_fuse_comm_list = [
            [0, 2, 16, 18],
            [1, 3, 17, 19],
            [4, 6, 20, 22],
            [5, 7, 21, 23],
            [8, 10, 24, 26],
            [9, 11, 25, 27],
            [12, 14, 28, 30],
            [13, 15, 29, 31],
        ]
        pp_mp_fuse_comm_list = [
            [0, 1, 8, 9],
            [2, 3, 10, 11],
            [4, 5, 12, 13],
            [6, 7, 14, 15],
            [16, 17, 24, 25],
            [18, 19, 26, 27],
            [20, 21, 28, 29],
            [22, 23, 30, 31],
        ]

        np.testing.assert_array_equal(
            sorted(dp_sep_fuse_comm_list),
            sorted(topo.get_fused_ranks(["dp", "sep"])),
        )
        np.testing.assert_array_equal(
            sorted(pp_mp_fuse_comm_list),
            sorted(topo.get_fused_ranks(["pp", "mp"])),
        )

        # test get_hybrid_group_names
        parallel_names = ["dp", "pp", "sharding", "sep", "mp"]
        np.testing.assert_array_equal(
            parallel_names, topo.get_hybrid_group_names()
        )

        # test get_dims
        np.testing.assert_array_equal(2, topo.get_dim("dp"))
        np.testing.assert_array_equal(2, topo.get_dim("mp"))
        np.testing.assert_array_equal(2, topo.get_dim("pp"))
        np.testing.assert_array_equal(2, topo.get_dim("sharding"))
        np.testing.assert_array_equal(2, topo.get_dim("sep"))

        # test world size
        self.assertEqual(topo.world_size(), 32)

        # test get_rank
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, sep=0, mp=0), 0)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, sep=0, mp=1), 1)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, sep=1, mp=0), 2)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=0, sep=1, mp=1), 3)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, sep=0, mp=0), 4)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, sep=0, mp=1), 5)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, sep=1, mp=0), 6)
        self.assertEqual(topo.get_rank(dp=0, pp=0, sharding=1, sep=1, mp=1), 7)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, sep=0, mp=0), 8)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, sep=0, mp=1), 9)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, sep=1, mp=0), 10)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=0, sep=1, mp=1), 11)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, sep=0, mp=0), 12)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, sep=0, mp=1), 13)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, sep=1, mp=0), 14)
        self.assertEqual(topo.get_rank(dp=0, pp=1, sharding=1, sep=1, mp=1), 15)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, sep=0, mp=0), 16)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, sep=0, mp=1), 17)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, sep=1, mp=0), 18)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=0, sep=1, mp=1), 19)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, sep=0, mp=0), 20)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, sep=0, mp=1), 21)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, sep=1, mp=0), 22)
        self.assertEqual(topo.get_rank(dp=1, pp=0, sharding=1, sep=1, mp=1), 23)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, sep=0, mp=0), 24)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, sep=0, mp=1), 25)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, sep=1, mp=0), 26)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=0, sep=1, mp=1), 27)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, sep=0, mp=0), 28)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, sep=0, mp=1), 29)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, sep=1, mp=0), 30)
        self.assertEqual(topo.get_rank(dp=1, pp=1, sharding=1, sep=1, mp=1), 31)

        # test get_coord
        self.assertEqual(topo.get_coord(0), topo.coordinate(0, 0, 0, 0, 0))
        self.assertEqual(topo.get_coord(1), topo.coordinate(0, 0, 0, 0, 1))
        self.assertEqual(topo.get_coord(2), topo.coordinate(0, 0, 0, 1, 0))
        self.assertEqual(topo.get_coord(3), topo.coordinate(0, 0, 0, 1, 1))
        self.assertEqual(topo.get_coord(4), topo.coordinate(0, 0, 1, 0, 0))
        self.assertEqual(topo.get_coord(5), topo.coordinate(0, 0, 1, 0, 1))
        self.assertEqual(topo.get_coord(6), topo.coordinate(0, 0, 1, 1, 0))
        self.assertEqual(topo.get_coord(7), topo.coordinate(0, 0, 1, 1, 1))
        self.assertEqual(topo.get_coord(8), topo.coordinate(0, 1, 0, 0, 0))
        self.assertEqual(topo.get_coord(9), topo.coordinate(0, 1, 0, 0, 1))
        self.assertEqual(topo.get_coord(10), topo.coordinate(0, 1, 0, 1, 0))
        self.assertEqual(topo.get_coord(11), topo.coordinate(0, 1, 0, 1, 1))
        self.assertEqual(topo.get_coord(12), topo.coordinate(0, 1, 1, 0, 0))
        self.assertEqual(topo.get_coord(13), topo.coordinate(0, 1, 1, 0, 1))
        self.assertEqual(topo.get_coord(14), topo.coordinate(0, 1, 1, 1, 0))
        self.assertEqual(topo.get_coord(15), topo.coordinate(0, 1, 1, 1, 1))
        self.assertEqual(topo.get_coord(16), topo.coordinate(1, 0, 0, 0, 0))
        self.assertEqual(topo.get_coord(17), topo.coordinate(1, 0, 0, 0, 1))
        self.assertEqual(topo.get_coord(18), topo.coordinate(1, 0, 0, 1, 0))
        self.assertEqual(topo.get_coord(19), topo.coordinate(1, 0, 0, 1, 1))
        self.assertEqual(topo.get_coord(20), topo.coordinate(1, 0, 1, 0, 0))
        self.assertEqual(topo.get_coord(21), topo.coordinate(1, 0, 1, 0, 1))
        self.assertEqual(topo.get_coord(22), topo.coordinate(1, 0, 1, 1, 0))
        self.assertEqual(topo.get_coord(23), topo.coordinate(1, 0, 1, 1, 1))
        self.assertEqual(topo.get_coord(24), topo.coordinate(1, 1, 0, 0, 0))
        self.assertEqual(topo.get_coord(25), topo.coordinate(1, 1, 0, 0, 1))
        self.assertEqual(topo.get_coord(26), topo.coordinate(1, 1, 0, 1, 0))
        self.assertEqual(topo.get_coord(27), topo.coordinate(1, 1, 0, 1, 1))
        self.assertEqual(topo.get_coord(28), topo.coordinate(1, 1, 1, 0, 0))
        self.assertEqual(topo.get_coord(29), topo.coordinate(1, 1, 1, 0, 1))
        self.assertEqual(topo.get_coord(30), topo.coordinate(1, 1, 1, 1, 0))
        self.assertEqual(topo.get_coord(31), topo.coordinate(1, 1, 1, 1, 1))

        # test get_axis_list
        self.assertEqual(
            topo.get_axis_list("dp", 0),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )
        self.assertEqual(
            topo.get_axis_list("dp", 1),
            [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        )

        self.assertEqual(
            topo.get_axis_list("sep", 0),
            [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29],
        )
        self.assertEqual(
            topo.get_axis_list("sep", 1),
            [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31],
        )

        self.assertEqual(
            topo.get_axis_list("mp", 0),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        )
        self.assertEqual(
            topo.get_axis_list("mp", 1),
            [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        )
        self.assertEqual(
            topo.get_axis_list("pp", 0),
            [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23],
        )
        self.assertEqual(
            topo.get_axis_list("pp", 1),
            [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31],
        )
        self.assertEqual(
            topo.get_axis_list("sharding", 0),
            [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
        )
        self.assertEqual(
            topo.get_axis_list("sharding", 1),
            [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31],
        )

        # test get_dim_size
        self.assertEqual(topo.get_dim_size("dp"), 2)
        self.assertEqual(topo.get_dim_size("mp"), 2)
        self.assertEqual(topo.get_dim_size("pp"), 2)
        self.assertEqual(topo.get_dim_size("sharding"), 2)
        self.assertEqual(topo.get_dim_size("sep"), 2)


if __name__ == '__main__':
    unittest.main()
