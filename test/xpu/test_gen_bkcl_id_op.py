# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
from multiprocessing import Process

from launch_function_helper import _find_free_port, wait

os.environ['GLOG_vmodule'] = "gen_bkcl_id_op*=10,gen_comm_id*=10"

import paddle
from paddle.base import core

paddle.enable_static()


def run_gen_bkc_id(attr):
    bkcl_comm_num = attr['bkcl_comm_num']
    use_hallreduce = attr['use_hierarchical_allreduce']

    startup_program = paddle.static.default_startup_program()
    main_program = paddle.static.default_main_program()

    with paddle.static.program_guard(main_program, startup_program):
        bkcl_id_var = startup_program.global_block().create_var(
            name="BKCLID", persistable=True, type=core.VarDesc.VarType.RAW
        )

        for i in range(1, bkcl_comm_num):
            startup_program.global_block().create_var(
                name=f"BKCLID_{i}",
                persistable=True,
                type=core.VarDesc.VarType.RAW,
            )

        if use_hallreduce:
            for i in range(0, bkcl_comm_num):
                startup_program.global_block().create_var(
                    name=f"Hierarchical_inter_BKCLID_{i}",
                    persistable=True,
                    type=core.VarDesc.VarType.RAW,
                )
                startup_program.global_block().create_var(
                    name=f"Hierarchical_exter_BKCLID_{i}",
                    persistable=True,
                    type=core.VarDesc.VarType.RAW,
                )

        startup_program.global_block().append_op(
            type="gen_bkcl_id",
            inputs={},
            outputs={"BKCLID": bkcl_id_var},
            attrs=attr,
        )

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    exe.run(startup_program)


class TestGenBKCLIdOp(unittest.TestCase):
    def setUp(self):
        try:
            self._dist_ut_port_0 = int(os.environ["PADDLE_DIST_UT_PORT"])
        except Exception as e:
            self._dist_ut_port_0 = _find_free_port(set())

    def gen_bkcl_id(self, nranks=2):
        bkcl_comm_num = 1
        if nranks == 2:
            use_hallreduce = False
            hallreduce_inter_nranks = -1
        elif nranks == 4:
            use_hallreduce = True
            hallreduce_inter_nranks = 2

        port = self._dist_ut_port_0
        trainers = []
        for i in range(nranks):
            trainers.append(f'127.0.0.1:{port + i}')

        attr = {
            "trainers": trainers,
            "trainer_id": 0,
            "bkcl_comm_num": bkcl_comm_num,
            "use_hierarchical_allreduce": use_hallreduce,
            "hierarchical_allreduce_inter_nranks": hallreduce_inter_nranks,
        }

        procs = []
        for i in range(nranks):
            attr['trainer_id'] = i
            # NOTE: multiprocessing cannot be covered by coverage
            p = Process(target=run_gen_bkc_id, args=(attr,))
            p.start()
            procs.append(p)

        wait(procs, timeout=120)

    def test_flat(self):
        print(">>> test gen flat bkcl id")
        self.gen_bkcl_id(2)
        print("<<< end test gen flat bkcl id")
        print()

    def test_hierarchical(self):
        print(">>> test gen hierarchical bkcl id")
        self.gen_bkcl_id(4)
        print("<<< end test gen hierarchical bkcl id")


if __name__ == "__main__":
    unittest.main()
