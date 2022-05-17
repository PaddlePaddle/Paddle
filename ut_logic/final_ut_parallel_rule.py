# -*- coding: utf-8 -*-
import os
import time
import json
import datetime
import codecs


# 单独跑出 ctest -N 所有的单测：耗时 + 显存 信息。存储在 testlog 目录中

# 从 testlog 中抽取 单测 + 耗时 + 显存 信息
def get_all_ut_mem_time():
    filedir = '/zhangbo/testlog/'
    all_ut_time_mem_dic = {}
    for parent, dirs, files in os.walk(filedir):
        for f in files:
            if f.endswith('$-gpu.log'):
                continue
            case1 = f.replace('^', '').replace('$.log', '')
            all_ut_time_mem_dic[case1] = {}
            filename = '%s%s' %(parent, f)
            fi= codecs.open(filename, 'r',encoding= u'utf-8',errors='ignore')
            lines = fi.readlines()
            mem1 = -1
            mem_reserved1 = -1
            mem_nvidia1 = -1
            caseTime = -1
            for line in lines: 
                if '[max memory reserved] gpu' in line:
                    mem_reserved = round(float(line.split('[max memory reserved] gpu')[1].split(':')[1].split('\\n')[0].strip()),2)
                    if mem_reserved > mem_reserved1:
                        mem_reserved1 = mem_reserved
                if 'MAX_GPU_MEMORY_USE=' in line:
                    mem_nvidia = round(float(line.split('MAX_GPU_MEMORY_USE=')[1].split('\\n')[0].strip()),2)
                    if mem_nvidia > mem_nvidia1:
                        mem_nvidia1 = mem_nvidia
                if 'Total Test time (real)' in line:
                    caseTime = float(line.split('Total Test time (real) =')[1].split('sec')[0].strip())

            if mem_reserved1 != -1 :
                all_ut_time_mem_dic[case1]['mem_reserved'] = mem_reserved1
            if mem_nvidia1 != -1:
                all_ut_time_mem_dic[case1]['mem_nvidia'] = mem_nvidia1        
            if caseTime != -1:
                all_ut_time_mem_dic[case1]['time'] = caseTime
        
        with open("/zhangbo/ut_logic/ut_mem_time.json", "w") as f:
            json.dump(all_ut_time_mem_dic, f)
        f.close()

# 筛选出所有 mem_nvidia 为 0 的
def get_ut_mem0():
    with open("/zhangbo/ut_logic/ut_mem_time.json", 'r') as load_f:
        tests_mem_time = json.load(load_f)
    case_dict = {}
    for case in tests_mem_time:
        if 'mem_nvidia' in tests_mem_time[case]:
            if tests_mem_time[case]["mem_nvidia"] == 0 :
                if 'mem_0' not in case_dict:
                    case_dict['mem_0'] = []
                case_dict['mem_0'].append(case)
    
    with open("/zhangbo/ut_logic/ut_mem_0.json","w") as f:
        json.dump(case_dict, f)
    load_f.close()
    f.close()

# 生成不占显存的高并发单测执行文件：
def generate_mem0_parallel_ut_file():
    # 所有mem=0的名单：
    with open("/zhangbo/ut_logic/ut_mem_0.json", 'r') as load_f:
        tests_mem_0_map = json.load(load_f)
    # 低并发名单：
    with open("/zhangbo/ut_logic/ut_mem_0_low_parallel.json", 'r') as load_f1:
        tests_mem_0_low_parallel_map = json.load(load_f1)
    # 从mem0中去除低并发名单中的单测
    count = 0
    high_parallel_job = '^job$'
    for case in tests_mem_0_map['mem_0']:
        if (case not in tests_mem_0_low_parallel_map["mem_0_timeout"]):
            high_parallel_job = high_parallel_job + '|^' + case + '$'
            count = count + 1

    load_f.close()
    load_f1.close()
    print("{}".format(high_parallel_job))
    
if __name__ == '__main__':
    get_all_ut_mem_time()
    get_ut_mem0()
    generate_mem0_parallel_ut_file()
