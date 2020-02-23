#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket
from flask import request
import random

random.seed(10)

app = Flask(__name__, static_url_path="")

job_manager = JobInfoManager()


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


class JobInfoManager(object):
    def __init__(self):
        self.job["test_job_id_1234"] = self._init_job_pods()
        self._t_id = None
        self._lock = threading.Lock()

    def _init_job_pods(self):
        #job = {"job_id": "test_job_id_1234", "pods": []}
        pods = []
        port = 7070
        for i in range(0, 16, 2):
            pod = {}
            pod["pod_id"] = i / 2
            pod["running"] = True
            pod["addr"] = "127.0.0.1"
            pod["pod_port"] = port + i
            pod["trainer_ports"] = [port + i + 1]
            pods.append(pod)
        #job["pods"] = pods
        return pods

    def start(self):
        assert self._t_id is None, "thread has been started"

        thread = Thread(target=self.run)
        thread.start()
        return str(thread)

    def get_job_pods(self, job_id):
        with self._lock:
            return self.jobs[job_id]

    def del_tail(self, pods_num):
        with self._lock:
            pods = self.job["pods"]
            assert pods_num < len(pods), "can't delete pods_num:%d".format(
                pods_num)
            #return pods[:-pods_num]

    def add_tail(self, pods_num):
        pass

    def run(self):
        while (True):
            with self._lock:
                generate_job_pods()
            time.sleep(300)


@app.route('/rest/1.0/get/query_pods', methods=['GET'])
def get_job_pods():
    try:
        job_id = request.args.get('job_id')
        print("job_id:", job_id)
    except:
        return jsonify({'job_id': {}})
    job_pods = job_manager.get_job_pods(job_id)
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
