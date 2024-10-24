# Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

import enum

from .xgb_cost_model import XgbCostModel


class CostModelType(enum.Enum):
    XGB = 1


class CostModel:
    """
    A base class to call different cost model algorithm.
    """

    def __init__(self, model_type=CostModelType.XGB):
        """
        Constructor
        """
        self.model = None
        if model_type == CostModelType.XGB:
            self.model = XgbCostModel()
        else:
            raise ValueError("Illegal CostModelType")

    def train(self, samples, labels):
        """
        Train the model.

        Args:
            samples(list|numpy): an array of numpy array representing a batch
                of input samples.
            labels(list|numpy): an array of float representing a batch of labels
        """
        return self.model.train(samples, labels)

    def predict(self, samples):
        """
        Predict

        Args:
            samples(list|numpy): an array of numpy array representing a batch
                of input samples.
        Returns:
            np.array representing labels
        """
        return self.model.predict(samples)

    def save(self, path):
        """
        Save the trained model.

        Args:
            path(str): path to save
        """
        return self.model.save(path)

    def load(self, path):
        """
        Load the model

        Args:
            path(str): path to load
        """
        return self.model.load(path)

    def update(self, samples, labels):
        # TODO
        pass
