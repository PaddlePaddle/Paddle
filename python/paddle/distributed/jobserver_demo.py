#!flask/bin/python
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket

app = Flask(__name__, static_url_path="")


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            print("socket name: %s" % s.getsockname()[1])
            return s.getsockname()[1]

    port_set = {}
    step = 0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set[port] = ""

        if len(port_set) >= num:
            return port_set

        step += 1
        if step > 100:
            print("can't find avilable port")
            return None

    return None


@app.route('/todo/api/v1.0/ports', methods=['GET'])
def get_ports():
    try:
        ports = request.args.get('port_num')
        ports = int(ports)
        if ports > 32:
            ports = 32
    except:
        return jsonify({'ports': {}})
    return jsonify({'ports': find_free_ports(ports)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
