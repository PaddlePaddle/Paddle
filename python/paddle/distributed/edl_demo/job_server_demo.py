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

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

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
        self._ip_podnum = None

    def _make_job_pods(self, ip_podnum, step_id=0):
        """
        ip_podnum:[(ip,pod_num),(ip, pod_num)]
        """
        assert type(node_ips) == list, "{} must be a list".format(node_ips)
        pods = []
        pod_rank = 0
        for n_p in ip_podnum:
            ip = n_p[0]
            pod_num = n_p[1]
            port = 6070
            for i in range(0, pod_num):
                pod = {}
                pod["pod_id"] = "pod_{}_{}_{}".format(pod_rank, 0, step_id)
                pod["running"] = True
                pod["addr"] = ip

                pod["pod_port"] = port
                pod["trainer_ports"] = [port + 1]

                pods.append(pod)
                port += 2
                pod_rank += 1

        return pods

    def start(self, node_ips):
        assert self._t_id is None, "thread has been started"

        self._ip_list = ['127.0.0.1']
        if node_ips is not None:
            self._ip_list = [x.strip() for x in node_ips.split(',')]

        self._ip_podnum = []
        for ip in self._ip_list:
            self._ip_podnum.append((ip, 8))

        pods = self._make_job_pods(ip_podnum)
        self.job[self._job_id] = pods

        thread = threading.Thread(target=self.run)
        self._t_id = thread.start()
        #print("job manager started!")

    def get_job_pods(self, job_id):
        with self._lock:
            return self.job[job_id]

    def _change(self, job_id, node_rank, pods_num, step_id):
        with self._lock:
            self._ip_podnum[node_rank][1] = pods_num
            pods = _make_job_pods(self._ip_podnum)
            self.job[_job_id] = pods
            print("changed pods {}".format(pods))

    def run(self):
        step_id = 0
        modify = True
        while (True):
            time.sleep(15)
            if modify:
                step_id += 1
                print("del 2 pods")
                self._change(self._job_id, len(self._ip_list) - 1, 6, step_id)
                time.sleep(15)

                print("add 2 pods")
                self._change(self._job_id, len(self._ip_list) - 1, 8, step_id)
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
