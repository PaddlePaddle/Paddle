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
    args = parser.parse_args()
    return args


def process_job_log(log_data, device_id, log_start_time):
    log_pattern = r'.*?Profiler Info: Job \((\d+)\), type = (\w+), micro_batch_id = (\d+), job_start_time = (\d+.\d+), job_end_time = (\d+.\d+)'
    matches = re.findall(log_pattern, log_data)
    events = []
    last_end_time = None

    for match in matches:
        job_id, job_type, micro_batch_id, job_start_time, job_end_time = match
        if job_type in ["lr"]:
            continue

        start_time = float(job_start_time.strip()) * 1000
        end_time = float(job_end_time.strip()) * 1000

        if log_start_time > start_time:
            continue

        event_start = {
            "name": job_type + "_" + str(job_id),
            "cat": job_type,
            "ph": "B",
            "ts": start_time,
            "pid": 0,
            "tid": "GPU" + str(device_id),
            "cname": color_map[job_type],
        }
        event_end = {
            "name": job_type + "_" + str(job_id),
            "cat": job_type,
            "ph": "E",
            "pid": 0,
            "ts": end_time,
            "tid": "GPU" + str(device_id),
            "cname": color_map[job_type],
        }
        events.append(event_start)
        events.append(event_end)

        last_end_time = end_time

    return events


def process_step_log(log_data, device_id):
    start_pattern = r'.*?NVTX range push: (\d+), time: (\d+.\d+)'
    end_pattern = r'.*?NVTX range pop, time: (\d+.\d+)'
    start_matches = re.findall(start_pattern, log_data)
    end_matches = re.findall(end_pattern, log_data)
    end_matches = end_matches[len(end_matches) - len(start_matches) :]

    step_info = []
    for start_match, stop_match in zip(start_matches, end_matches):
        step_id, start_time = start_match
        stop_time = stop_match
        if int(step_id) >= len(step_info):
            for _ in range(int(step_id) - len(step_info) + 1):
                step_info.append([float('inf'), 0])
        step_info[int(step_id)] = [start_time, stop_time]

    start_step = 0
    for info in step_info:
        if info[0] == float('inf'):
            start_step += 1
    return step_info, start_step


def main():
    args = parse_args()
    all_events = []
    step_infos = []
    for device_id in args.devices.split(","):
        _logger.info(f"Process device {device_id}")
        device_id = int(device_id)
        log_file = os.path.join(args.log_dir, "workerlog." + str(device_id))
        with open(log_file, "r") as f:
            log_data = f.read()

        step_info, start_step = process_step_log(log_data, device_id)

        if len(step_info) > len(step_infos):
            for _ in range(len(step_info) - len(step_infos)):
                step_infos.append([float('inf'), 0])
        for i, info in enumerate(step_info):
            if info[0] == float('inf'):
                continue
            start_time = float(info[0].strip()) * 1000
            stop_time = float(info[1].strip()) * 1000

            step_infos[i][0] = min(step_infos[i][0], start_time)
            step_infos[i][1] = max(step_infos[i][1], stop_time)

        events = process_job_log(log_data, device_id, step_infos[start_step][0])
        all_events.extend(events)

    for i, info in enumerate(step_infos):
        if info[0] == float('inf'):
            continue
        start_time = info[0]
        if i > 0:
            start_time = max(start_time, step_infos[i - 1][1])
        event_start = {
            "name": "step" + str(i),
            "cat": "step",
            "ph": "B",
            "ts": start_time,
            "pid": 0,
            "tid": "Step",
            "cname": color_map["default"],
        }
        event_end = {
            "name": "step" + str(i),
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
    for i in range(len(args.devices.split(","))):
        all_events.extend(
            [
                {
                    "args": {"name": f"GPU:{i}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 0,
                    "tid": i + 2334,
                    "ts": 0,
                }
            ]
        )
    json_str = json.dumps({"traceEvents": all_events})
    for i in range(len(args.devices.split(","))):
        json_str = json_str.replace('"Step"', '2333')
        json_str = json_str.replace(f'"GPU{i}"', f'{i + 2334}')

    with open(save_path, "w") as f:
        f.write(json_str)
    _logger.info(f"Save pipeline profile to {save_path}")


if __name__ == "__main__":
    main()
