#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket
from flask import request
import random
import threading
import time

random.seed(10)

app = Flask(__name__, static_url_path="")


class JobInfoManager(object):
    def __init__(self):
        self.job = {}
        self._t_id = None
        self._lock = threading.Lock()
        self._last_port = None

        pods = self._init_job_pods()
        self._job_id = "test_job_id_1234"
        self.job[self._job_id] = pods

    def _init_job_pods(self):
        pods = []
        port = 7070
        for i in range(0, 16, 2):
            pod = {}
            pod["pod_id"] = "pod_{}_{}".format(i / 2, 0)
            pod["running"] = True
            pod["addr"] = "127.0.0.1"
            pod_port = port + i
            pod["pod_port"] = pod_port
            pod["trainer_ports"] = [pod_port + 1]
            pods.append(pod)
            self._last_port = pod_port + 1
        return pods

    def start(self):
        assert self._t_id is None, "thread has been started"

        thread = threading.Thread(target=self.run)
        self._t_id = thread.start()
        print("job manager started!")

    def get_job_pods(self, job_id):
        with self._lock:
            return self.job[job_id]

    def del_tail(self, job_id, pods_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            assert pods_num < len(pods), "can't delete pods_num:%d".format(
                pods_num)
            self.job[job_id] = pods[:-pods_num]
            print("deleted pods {}".format(pods))

    def add_tail(self, job_id, pods_num, step_id):
        with self._lock:
            pods = self.job[job_id]
            pod_rank = len(pods)
            for i in range(0, pods_num * 2, 2):
                pod = {}
                pod["pod_id"] = "pod_{}_{}".format(pod_rank, step_id)
                pod["running"] = True
                pod["addr"] = "127.0.0.1"
                pod_port = self._last_port + 1
                pod["pod_port"] = pod_port
                pod["trainer_ports"] = [pod_port + 1]
                pods.append(pod)
                self._last_port = pod_port + 1

                pods.append(pod)
                pod_rank = len(pods)
            print("added pods {}".format(pods))

    def run(self):
        step_id = 0
        modify = True
        while (True):
            time.sleep(300)
            if modify:
                step_id += 1
                """
                print("del 1 pods")
                self.del_tail(self._job_id, 1, step_id)
                time.sleep(300)
                print("add 1 pods")
                self.add_tail(job_id, 1, step_id)
                modify = False
                """


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
        print("job_id:", job_id)
    except:
        return jsonify({'job_id': {}})
    job_pods = job_manager.get_job_pods("test_job_id_1234")
    print("job_pods:", job_pods)
    return jsonify({'job_id': "job_id", "pods": job_pods})


@app.route('/rest/1.0/post/job_runtime_static', methods=['POST'])
def update_job_static():
    if not request.json or not 'job_id' in request.json or not 'estimated_run_time' in request.json:
        abort(400)
    print(requests.json)


if __name__ == '__main__':
    job_manager.start()
    app.run(host='0.0.0.0', port=8180)
