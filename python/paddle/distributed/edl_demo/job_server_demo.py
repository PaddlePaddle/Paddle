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
from .. import utils

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(utils.add_arguments, argparser=parser)

add_arg(
    'node_ips', str, None,
    "Nodes's IP list, splitted by ','. For example, 192.168.0.1,192.168.0.2")

random.seed(10)

app = Flask(__name__, static_url_path="")


class JobInfoManager(object):
    def __init__(self):
        self.job = {}
        self._t_id = None
        self._lock = threading.Lock()

        self._job_id = "test_job_id_1234"
        self.job[self._job_id] = []
        self._ip_list = None
        self._ip_pod_num = None

    def _make_new_pods(self, start_rank, pod_num, step_id=0):
        pods = []
        for i in range(0, pod_num):
            pod = {}
            pod["pod_id"] = "pod_{}_{}".format(start_rank + i, step_id)
            pods.append(pod)

        return pods

    def _assign_pods_to_nodes(self, pods, ip_pod_num):
        """
        ip_podnum:[[ip,pod_num],[ip, pod_num]]
        """

        pod_rank = 0
        for node_rank, n_p in enumerate(ip_pod_num):
            ip = n_p[0]
            pod_num = n_p[1]
            port = 6070
            for i in range(0, pod_num):
                if pod_rank >= len(pods):
                    return

                pod = pods[pod_rank]
                #pod["pod_id"] = "pod_{}_{}".format(pod_rank, step_id)
                pod["running"] = True
                pod["addr"] = ip

                pod["pod_port"] = port
                pod["trainer_ports"] = [port + 1]

                port += 2
                pod_rank += 1

    def start(self, node_ips):
        assert self._t_id is None, "thread has been started"

        self._ip_list = ['127.0.0.1']
        if node_ips is not None:
            self._ip_list = [x.strip() for x in node_ips.split(',')]

        self._ip_pod_num = []
        pod_num = 0
        for ip in self._ip_list:
            self._ip_pod_num.append([ip, 8])
            pod_num += 8

        pods = self._make_new_pods(0, pod_num)
        self._assign_pods_to_nodes(pods, self._ip_pod_num)
        self.job[self._job_id] = pods

        thread = threading.Thread(target=self.run)
        self._t_id = thread.start()
        #print("job manager started!")

    def get_job_pods(self, job_id):
        with self._lock:
            return self.job[job_id]

    def _del_tail(self, job_id, pod_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            assert pod_num < len(pods), "can't delete pod_num:%d".format(
                pod_num)
            self.job[job_id] = pods[:-pod_num]
            #print("deleted pods {}".format(self.job[job_id]))

    def _add_tail(self, job_id, pod_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            start_rank = len(pods)
            new_pods = self._make_new_pods(start_rank, pod_num, step_id)
            pods.extend(new_pods)

            self._assign_pods_to_nodes(pods, self._ip_pod_num)
            self.job[self._job_id] = pods
            #print("added pods {}".format(pods))

    def run(self):
        step_id = 0
        modify = True
        while (True):
            time.sleep(30)  # 20minutes
            if modify:
                step_id += 1
                #print("del 2 pods")
                self._del_tail(self._job_id, 2, step_id)
                time.sleep(15)

                step_id += 1
                #print("add 2 pods")
                self._add_tail(self._job_id, 2, step_id)
                modify = False


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
    job_pods = job_manager.get_job_pods("test_job_id_1234")
    #print("job_pods:", job_pods)
    return jsonify({'job_id': job_id, "pods": job_pods})


@app.route('/rest/1.0/post/job_runtime_static', methods=['POST'])
def update_job_static():
    if not request.json or not 'job_id' in request.json or not 'estimated_run_time' in request.json:
        abort(400)
    #print(requests.json)


if __name__ == '__main__':
    args = parser.parse_args()
    job_manager.start(args.node_ips)
    app.run(host='0.0.0.0', port=8180)
