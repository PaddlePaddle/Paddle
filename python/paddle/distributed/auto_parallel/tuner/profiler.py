"""
parse cmd args & setting

load ctx restore env 


run 

write result
"""
import os
import argparse
import pickle
import numpy as np

import paddle
from paddle.fluid.framework import Program, _current_expected_place
from paddle.distributed.auto_parallel.process_group import clear_all_process_groups, get_all_process_groups, new_process_group 
from paddle.distributed.collective import _get_global_env

paddle.enable_static()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile_start_step",
        default=10,
        type=int,
        help="integer indicates the warmup step before starting profile." )
    parser.add_argument(
        "--profile_end_step",
        default=30,
        type=int,
        help="integer indicates at the end step of profile." )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="the rank id of the this process." )
    parser.add_argument(
        "--device_id",
        type=int,
        required=True,
        help="the device id of the this process." )
    parser.add_argument(
        "--ctx_filename",
        type=str,
        required=True,
        help="the filename to the profile context file saved by optimizaiton tuner" )
    parser.add_argument(
        "--main_program_filename",
        type=str,
        required=True,
        help="the filename to the main program decs file saved by optimizaiton tuner" )
    parser.add_argument(
        "--startup_program_filename",
        type=str,
        required=True,
        help="the filename to the startup program decs file saved by optimizaiton tuner" )

    args = parser.parse_args()

    return args

def init_process_groups(group_map, rank):
    for group_id, ranks in group_map.items():
        if group_id == 0 :
            continue
        new_process_group(ranks = ranks, group_id = group_id)
    
    # TODO should instantiate global group first
    all_process_groups = get_all_process_groups()
    for process_group in all_process_groups:
        if process_group.id == 0 or rank not in process_group.ranks :
            continue
        print(process_group)
        process_group.instantiate()

def gen_data(batch_size):
    sequence_len = 512
    tokens = []
    position_ids = []
    attention_mask = []
    labels = []
    loss_mask = []
    for _ in range(batch_size):
        tokens.append(np.random.randint(50304, size=sequence_len))
        position_ids.append(np.arange(sequence_len))
        attention_mask.append([np.tril(np.ones(sequence_len))])
        labels.append(np.random.randint(50304, size=sequence_len))
        loss_mask.append(np.ones(sequence_len))

    return tokens, position_ids, attention_mask, labels, loss_mask

def profiler():

    args = parse_args()

    # load ctx
    if not os.path.isfile(args.ctx_filename):
        raise ValueError("There is no profile context named {}.".format(args.ctx_filename))
    if not os.path.isfile(args.main_program_filename):
        raise ValueError("There is no main program named {}.".format(args.ctx_filename))
    if not os.path.isfile(args.startup_program_filename):
        raise ValueError("There is no startup program named {}.".format(args.ctx_filename))

    with open(args.ctx_filename, 'rb') as f:
        profile_ctx = pickle.load(f, encoding='latin1')    

    # TODO check type and exist
    print("=====profile" * 8)
    dist_env = profile_ctx['distributed_env']
    genv = _get_global_env() 
    genv = dist_env
    env = _get_global_env() 
    print(env.current_endpoint, env.device_id, env.rank)
    group_map = profile_ctx['group_map']
    init_process_groups(group_map, args.rank)
    print(dist_env.rank, dist_env.device_id, dist_env.trainer_endpoints)
    with open(args.main_program_filename, "rb") as f:
        main_program_desc_str = f.read()
        main_program = Program.parse_from_string(main_program_desc_str)
    with open(args.startup_program_filename, "rb") as f:
        startup_program_desc_str = f.read()
        startup_program = Program.parse_from_string(startup_program_desc_str)
    print("=====profile" * 8)

    # reconstruct communicator

    # build fake dataloader

    place_type = _current_expected_place()
    if not isinstance(place_type, paddle.CUDAPlace):
        raise NotImplementedError

    place = paddle.CUDAPlace(dist_env.device_id)
    print(dist_env.device_id)
    print(place)
    exe = paddle.static.Executor(place)

    exe.run(startup_program)

    # profile main
    eval_step = 0
    while eval_step < args.profile_end_step + 1:
        tokens, position_ids, attention_mask, labels, loss_mask = gen_data(1)
        loss = exe.run(main_program, 
                feed={
                    "tokens": tokens,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "loss_mask": loss_mask
                },
                fetch_list=['tmp_26'])
        print("step: %d, loss_print: %f" % (eval_step, loss[0]))
        eval_step += 1

if __name__ == "__main__":
    profiler()



    # /usr/bin/python3 -u -m paddle.distributed.auto_parallel.tuner.profile --rank 0 --device_id 0 --ctx_filename ./Sharing_stage_2_trial.pfcontext --main_program_filename ./Sharing_stage_2_trial.main_program_decs.0 --startup_program_filename ./Sharing_stage_2_trial.startup_program_decs.0