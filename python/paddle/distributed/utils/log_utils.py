# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import logging


def get_logger(log_level, name="root"):

    logger = logging.getLogger(name)

    # Avoid printing multiple logs
    logger.propagate = False

    if not logger.handlers:
        log_handler = logging.StreamHandler()
        logger.setLevel(log_level)
        log_format = logging.Formatter(
            '[%(asctime)-15s] [%(levelname)8s] %(filename)s:%(lineno)s - %(message)s'
        )
        log_handler.setFormatter(log_format)
        logger.addHandler(log_handler)
    else:
        logger.setLevel(log_level)
    return logger
