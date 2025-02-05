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

import os
import queue
import sys
import threading
import time

taskQueue = queue.Queue()


def worker(fun):
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()


def threadPool(threadPoolNum):
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(
            target=worker,
            args={
                doFun,
            },
        )
        thread.daemon = True
        threadPool.append(thread)
    return threadPool


def get_h_file_md5(rootPath):
    h_cu_files = f'{rootPath}/tools/h_cu_files.log'
    f = open(h_cu_files)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        os.system(f'md5sum {line} >> {rootPath}/tools/h_cu_md5.log')


def insert_pile_to_h_file(rootPath):
    h_cu_files = f'{rootPath}/tools/h_cu_files.log'
    f = open(h_cu_files)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        func = line.replace('/', '_').replace('.', '_')
        os.system(f'echo "\n#ifndef _PRECISE{func.upper()}_\n" >> {line}')
        os.system(f'echo "#define _PRECISE{func.upper()}_" >> {line}')
        os.system(f'echo "\n#include <cstdio>\n" >> {line}')
        os.system(
            f'echo "__attribute__((constructor)) static void calledFirst{func}()\n{{" >> {line}'
        )
        os.system(
            f'echo \'    fprintf(stderr,"precise test map fileeee: %s\\\\n", __FILE__);\n}}\' >> {line}'
        )
        os.system(f'echo "\n#endif" >> {line}')


def add_simple_cxx_test(rootPath):
    variant_test_path = f'{rootPath}/paddle/utils/variant_test.cc'
    variant_test_cmakeflie_path = f'{rootPath}/paddle/utils/CMakeLists.txt'
    if os.path.exists(variant_test_path) and os.path.exists(
        variant_test_cmakeflie_path
    ):
        simple_test_path = f'{rootPath}/paddle/utils/simple_precision_test.cc'
        os.system(f'touch {simple_test_path}')
        os.system(f"echo '#include \"gtest/gtest.h\"\n' >> {simple_test_path}")
        os.system(
            f'echo "TEST(interface_test, type) {{ }}\n" >> {simple_test_path}'
        )
        os.system(f'echo "cc_test(" >> {variant_test_cmakeflie_path}')
        os.system(
            f'echo "  simple_precision_test" >> {variant_test_cmakeflie_path}'
        )
        os.system(
            f'echo "  SRCS simple_precision_test.cc" >> {variant_test_cmakeflie_path}'
        )
        os.system(f'echo "  DEPS gtest)\n" >> {variant_test_cmakeflie_path}')


def remove_pile_from_h_file(rootPath):
    h_cu_files = f'{rootPath}/tools/h_cu_files.log'
    f = open(h_cu_files)
    lines = f.readlines()
    count = 12
    for line in lines:
        line = line.strip()
        while count > 0:
            os.system(f"sed -i '$d' {line}")
            count = count - 1
        count = 12


def get_h_cu_file(file_path):
    rootPath = file_path[0]
    dir_path = file_path[1]
    filename = file_path[2]
    ut = filename.replace('^', '').replace('$', '').replace('.log', '')
    ut_path = f"{rootPath}/build/ut_map/{ut}"
    if os.path.exists(ut_path):
        os.system(
            f"cat {dir_path}/{filename} | grep 'precise test map fileeee:'| uniq >> {rootPath}/build/ut_map/{ut}/related_{ut}.txt"
        )
    else:
        print(f"{ut} has failed,no has directory")


def doFun(file_path):
    get_h_cu_file(file_path)


def main(rootPath, dir_path):
    """
    get useful message
    """
    startTime = int(time.time())
    test_h_cu_dict = {}
    pool = threadPool(23)
    for i in range(pool.__len__()):
        pool[i].start()
    files = os.listdir(dir_path)
    for filename in files:
        file_path = [rootPath, dir_path, filename]
        taskQueue.put(file_path)
    taskQueue.join()
    endTime = int(time.time())
    print('analy h/cu file cost Time: %s' % (endTime - startTime))


if __name__ == "__main__":
    func = sys.argv[1]
    if func == 'get_h_file_md5':
        rootPath = sys.argv[2]
        get_h_file_md5(rootPath)
    elif func == 'insert_pile_to_h_file':
        rootPath = sys.argv[2]
        insert_pile_to_h_file(rootPath)
        add_simple_cxx_test(rootPath)
    elif func == 'analy_h_cu_file':
        dir_path = sys.argv[2]
        rootPath = sys.argv[3]
        main(rootPath, dir_path)
    elif func == 'remove_pile_from_h_file':
        rootPath = sys.argv[2]
        remove_pile_from_h_file(rootPath)
