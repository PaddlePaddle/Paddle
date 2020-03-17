#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket
from flask import request
import random
import threading
import time
import argparse
import copy
import functools
import paddle.distributed.utils as utils

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utils.add_arguments, argparser=parser)

add_arg(
    'node_ips', str, None,
    "Nodes's IP list, splitted by ','. For example, 192.168.0.1,192.168.0.2")
add_arg('gpu_num_of_node', int, 8, "")
add_arg('pod_num_of_node', int, 1, "")
add_arg('del_pods_one_step', int, 1, "")
add_arg('add_pods_one_step', int, 1, "")
add_arg('time_interval_to_change', int, 900, "")

random.seed(10)

app = Flask(__name__, static_url_path="")


class JobInfoManager(object):
    def __init__(self):
        self.job = {}
        self.job_stage = {}

        self._t_id = None
        self._lock = threading.Lock()

        self._job_id = "test_job_id_1234"
        self.job[self._job_id] = []
        self.job_stage[self._job_id] = -1
        self._ip_list = None
        self._ip_pod_num = None
        self._gpu_num_of_node = None
        self._pod_num_of_node = None
        self._gpu_num_of_pod = None

    def _make_new_pods(self, start_rank, pod_num, step_id=0):
        pods = []
        for i in range(0, pod_num):
            pod = {}
            pod["pod_id"] = "pod_{}_{}".format(start_rank + i, step_id)
            pods.append(pod)

        return pods

    def _assign_pods_to_nodes(self, pods, ip_pod_num, gpu_num_of_pod):
        """
        ip_podnum:[[ip,pod_num],[ip, pod_num]]
        """

        pod_rank = 0
        for node_rank, n_p in enumerate(ip_pod_num):
            ip = n_p[0]
            pod_num = n_p[1]
            port = 8070
            for i in range(0, pod_num):
                if pod_rank >= len(pods):
                    return

                pod = pods[pod_rank]
                pod["running"] = True
                pod["addr"] = ip

                pod["pod_port"] = port
                port += 1

                pod["trainer_ports"] = []
                pod["gpus"] = []
                for j in range(0, gpu_num_of_pod):
                    pod["trainer_ports"].append(port)
                    pod["gpus"].append(i * gpu_num_of_pod + j)
                    port += 1
                pod_rank += 1

    def start(self, node_ips, gpu_num_of_node, pod_num_of_node):
        assert self._t_id is None, "thread has been started"

        self._ip_list = ['127.0.0.1']
        if node_ips is not None:
            self._ip_list = [x.strip() for x in node_ips.split(',')]

        self._pod_num_of_node = pod_num_of_node
        self._gpu_num_of_node = gpu_num_of_node
        assert gpu_num_of_node % pod_num_of_node == 0, "{} % {} must be 0.".format(
            gpu_num_of_node, pod_num_of_node)
        self._gpu_num_of_pod = gpu_num_of_node / pod_num_of_node

        self._ip_pod_num = []
        pod_num = 0
        for ip in self._ip_list:
            self._ip_pod_num.append([ip, pod_num_of_node])
            pod_num += pod_num_of_node

        pods = self._make_new_pods(0, pod_num)
        self._assign_pods_to_nodes(pods, self._ip_pod_num, self._gpu_num_of_pod)
        self.job[self._job_id] = pods

        thread = threading.Thread(target=self.run)
        self._t_id = thread.start()
        #print("job manager started!")

    def get_job_pods(self, job_id):
        with self._lock:
            return self.job[job_id], self.job_stage[job_id]

    def _del_tail(self, job_id, pod_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            assert pod_num < len(pods), "can't delete pod_num:%d".format(
                pod_num)
            self.job[job_id] = pods[:-pod_num]
            self.job_stage[self._job_id] = step_id
            #print("deleted pods {}".format(self.job[job_id]))

    def _add_tail(self, job_id, pod_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            start_rank = len(pods)
            new_pods = self._make_new_pods(start_rank, pod_num, step_id)
            pods.extend(new_pods)

            self._assign_pods_to_nodes(pods, self._ip_pod_num,
                                       self._gpu_num_of_pod)
            self.job[self._job_id] = pods
            self.job_stage[self._job_id] = step_id
            #print("added pods {}".format(pods))

    def run(self):
        step_id = -1
        modify = True
        while (True):
            time.sleep(args.time_interval_to_change)  # 20minutes
            if modify:
                step_id += 1
                #print("del 2 pods")
                self._del_tail(self._job_id, args.del_pods_one_step, step_id)
                time.sleep(args.time_interval_to_change)

                step_id += 1
                #print("add 2 pods")
                self._add_tail(self._job_id, args.add_pods_one_step, step_id)
                #modify = False


job_manager = JobInfoManager()


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/rest/1.0/get/query_pods', methods=['GET'])
def get_job_pods():
    try:
        job_id = request.args.get('job_id')
        #print("job_id:", job_id)
        job_id = "test_job_id_1234"
    except:
        return jsonify({'job_id': {}})
    job_pods, flag = job_manager.get_job_pods("test_job_id_1234")
    #print("job_pods:", job_pods)
    return jsonify({'job_id': job_id, "pods": job_pods, "job_stage_flag": flag})


@app.route('/rest/1.0/post/job_runtime_static', methods=['POST'])
def update_job_static():
    if not request.json or not 'job_id' in request.json or not 'estimated_run_time' in request.json:
        abort(400)
    #print(requests.json)


class ScopeKV(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._scope = {}

    def get_value(self, scope, key):
        with self._lock:
            if scope in self._scope and key in self._scope[scope]:
                return self._scope[scope][key]
        return None

    def post(self, scope, key, data):
        with self._lock:
            if scope not in self._scope[scope]:
                self._scope[scope] = {}
            self.job[scope][key] = data

    def get_scope(self, scope):
        with self._lock:
            if scope in self._scope[scope]:
                return self._scope[scope]
        return None


if __name__ == '__main__':
    args = parser.parse_args()
    #print("input_args:", args)
    job_manager.start(
        args.node_ips,
        gpu_num_of_node=args.gpu_num_of_node,
        pod_num_of_node=args.pod_num_of_node)
    app.run(host='0.0.0.0', port=8180, threaded=True, processes=1)
