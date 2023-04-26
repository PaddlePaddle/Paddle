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

import argparse

from flask import Flask, jsonify, request

app = Flask(__name__)
test_value = 0.66943359375


@app.route('/run/predict', methods=['POST'])
def echo():
    # Get the data from the request
    request_json = request.json
    # data = request_json['text']

    # Echo the data back in the response
    response = {'result': [str(test_value)]}

    # Return the response in JSON format
    return jsonify(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True, help='port')
    parser.add_argument(
        '--ip', type=str, required=False, default='localhost', help='ip'
    )
    args = parser.parse_args()
    app.run(host=args.ip, port=args.port)
