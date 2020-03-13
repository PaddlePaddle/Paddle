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
import multiprocessing

app = Flask(__name__, static_url_path="")


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


kv_store = ScopeKV()


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/rest/1.0/get/pod', methods=['GET'])
def get_value():
    try:
        scope = request.args.get('scope')
        key = request.args.get('scope')
        value = kv_store.get_value(scope, key)
    except:
        return jsonify({})
    #print("job_pods:", job_pods)
    return jsonify({'scope': scope, 'key': key, "value": data})


@app.route('/rest/1.0/get/scope', methods=['GET'])
def get_scope():
    try:
        scope = request.args.get('scope')
        value = kv_store.get_scope(scope)
    except:
        return jsonify({})
    return jsonify({'scope': scope, "value": data})


@app.route('/rest/1.0/post/pod', methods=['POST'])
def update_data():
    print(requests.json)
    try:
        scope = request.json["scope"]
        key = request.json["key"]
        value = request.json["value"]
    except:
        return jsonify({'invalid arguments'})
    kv_store.post(scope, key, value)


class HttpServer(object):
    def __init__(self):
        self._proc = None

    def _start(self, host, port):
        app.run(host=host, port=port, threaded=True, processes=1)

    def start(self, host, port):
        self._proc = multiprocessing.Process(
            target=self.start, args=(host, port))

    def stop(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()

    def is_avlive(self):
        if self._proc is None:
            return False

        if not self._proc.is_alive():
            return False

        return True


kv_server = HttpServer()

if __name__ == '__main__':
    start_server(host='0.0.0.0', port=8180)
