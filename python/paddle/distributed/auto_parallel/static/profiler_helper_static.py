# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import json
import logging
import os
import re
from argparse import ArgumentParser

import paddle
from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


color_map = {
    "forward": "thread_state_running",  # RGB: 126, 200, 148
    "backward": "rail_idle",  # RGB: 238, 142, 0
    "optimizer": "rail_response",  # RGB: 238, 142, 0
    "default": "thread_state_unknown",  # RGB: 199, 155, 125
}


def parse_args():
    parser = ArgumentParser()
    device_count = paddle.device.cuda.device_count()
    all_devices = ",".join([str(i) for i in range(device_count)])
    parser.add_argument("--devices", type=str, default=all_devices)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--multi_machine", action="store_true")
    args = parser.parse_args()
    return args


def process_job_log(log_data, device_id, multi_machine_idx=-1):
    log_pattern = r'.*?Profiler Info: Job \((\d+)\), type = (\w+), micro_batch_id = (\d+), job_start_time = (\d+.\d+), job_end_time = (\d+.\d+)'
    matches = re.findall(log_pattern, log_data)
    events = []
    last_end_time = None

    step_times = []
    step_start_time = 0
    step_end_time = 0

    start_job_type = ""

    for i, match in enumerate(matches):
        job_id, job_type, micro_batch_id, job_start_time, job_end_time = match

        if job_type != "default" and start_job_type == "":
            start_job_type = job_type

        start_time = float(job_start_time.strip()) * 1000
        end_time = float(job_end_time.strip()) * 1000

        is_start_time_recorded = 0

        if job_type == start_job_type and micro_batch_id == "0":
            if step_start_time != 0:
                step_times.append([step_start_time, step_end_time])
            step_start_time = start_time

        step_end_time = end_time

        tid_name = (
            "GPU" + str(device_id)
            if multi_machine_idx == -1
            else "GPU"
            + str(device_id)
            + "(machine:"
            + str(multi_machine_idx)
            + ")"
        )
        event_start = {
            "name": job_type + "_" + str(job_id),
            "cat": job_type,
            "ph": "B",
            "ts": start_time,
            "pid": 0,
            "tid": tid_name,
        }
        event_end = {
            "name": job_type + "_" + str(job_id),
            "cat": job_type,
            "ph": "E",
            "pid": 0,
            "ts": end_time,
            "tid": tid_name,
        }
        if job_type in color_map:
            event_start["cname"] = color_map[job_type]
            event_end["cname"] = color_map[job_type]

        events.append(event_start)
        events.append(event_end)

        last_end_time = end_time

    step_times.append([step_start_time, step_end_time])
    return events, step_times


def main():
    args = parse_args()
    all_events = []
    step_infos = []
    start_step = 0
    machine_num = 1

    def process_one_machine_log(log_dir, multi_machine_idx=-1):
        for device_id in args.devices.split(","):
            _logger.info(f"Process device {device_id}")
            device_id = int(device_id)
            log_file = os.path.join(log_dir, "workerlog." + str(device_id))
            with open(log_file, "r") as f:
                log_data = f.read()

            start_step_pattern = (
                r'.*?Schedule Profiler start at step (\d+) and end at step.*'
            )
            start_step_match = re.findall(start_step_pattern, log_data)
            start_step = (
                int(start_step_match[0]) if len(start_step_match) > 0 else 0
            )

            events, step_times = process_job_log(
                log_data, device_id, multi_machine_idx
            )
            all_events.extend(events)
            for i, info in enumerate(step_times):
                if len(step_infos) <= i:
                    step_infos.append([float("inf"), float("-inf")])
                step_infos[i][0] = min(step_infos[i][0], info[0])
                step_infos[i][1] = max(step_infos[i][1], info[1])

    if args.multi_machine:
        multi_machine_dirs = os.listdir(args.log_dir)
        multi_machine_dirs = [
            os.path.join(args.log_dir, d)
            for d in multi_machine_dirs
            if d.startswith("machine")
            and os.path.isdir(os.path.join(args.log_dir, d))
        ]
        machine_num = len(multi_machine_dirs)
        for i, d in enumerate(multi_machine_dirs):
            _logger.info(f"Process machine {i}")
            process_one_machine_log(d, i)
    else:
        process_one_machine_log(args.log_dir)

    for i, info in enumerate(step_infos):
        start_time = info[0]
        if i > 0:
            start_time = max(start_time, step_infos[i - 1][1])
        event_start = {
            "name": "step" + str(i + start_step),
            "cat": "step",
            "ph": "B",
            "ts": start_time,
            "pid": 0,
            "tid": "Step",
            "cname": color_map["default"],
        }
        event_end = {
            "name": "step" + str(i + start_step),
            "cat": "step",
            "ph": "E",
            "ts": info[1],
            "pid": 0,
            "tid": "Step",
            "cname": color_map["default"],
        }

        all_events.append(event_start)
        all_events.append(event_end)

    save_path = os.path.join(args.log_dir, "pipeline_profile.json")
    with open(save_path, "w") as f:
        f.write(json.dumps({"traceEvents": all_events}))
    _logger.info(f"Save pipeline profile to {save_path}")

    # support Perfetto format
    save_path = os.path.join(args.log_dir, "pipeline_profile_perfetto.json")
    all_events.extend(
        [
            {
                "args": {"name": "STEP"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": 2333,
                "ts": 0,
            }
        ]
    )

    for i in range(machine_num):
        for j in range(len(args.devices.split(","))):
            if machine_num > 1:
                name = f"GPU:{j}(machine:{i})"
                tid = i * len(args.devices.split(",")) + j + 2334
            else:
                name = f"GPU:{j}"
                tid = j + 2334
            all_events.extend(
                [
                    {
                        "args": {"name": name},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 0,
                        "tid": tid,
                        "ts": 0,
                    }
                ]
            )

    json_str = json.dumps({"traceEvents": all_events})
    json_str = json_str.replace('"Step"', '2333')

    for i in range(machine_num):
        for j in range(len(args.devices.split(","))):
            if machine_num > 1:
                json_str = json_str.replace(
                    f'"GPU{j}(machine:{i})"',
                    f'{i * len(args.devices.split(",")) + j + 2334}',
                )
            else:
                json_str = json_str.replace(f'"GPU{j}"', f'{j + 2334}')

    with open(save_path, "w") as f:
        f.write(json_str)
    _logger.info(f"Save pipeline profile to {save_path}")


if __name__ == "__main__":
    main()
