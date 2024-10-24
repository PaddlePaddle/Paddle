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

import numpy as np
import xgboost as xgb


class XgbCostModel:
    """
    A cost model implemented by XgbCostModel
    """

    def __init__(self):
        """
        Constructor
        """
        # Store the xgb.Booster, which is the output of xgb.train
        self.booster = None

        self.xgb_param = {}
        self.train_round = 10

    def train(self, samples, labels):
        """
        Train the model.

        Args:
            samples(list|numpy): an array of numpy array representing a batch
                of input samples.
            labels(list|numpy): an array of float representing a batch of labels

        Returns:
            xgb.Booster
        """
        lengths = [x.shape[0] for x in samples]
        if isinstance(samples, list):
            samples = np.concatenate(samples, axis=0)
        if isinstance(labels, list):
            labels = np.concatenate(
                [[y] * length for y, length in zip(labels, lengths)], axis=0
            )

        dmatrix = xgb.DMatrix(data=samples, label=labels)
        self.booster = xgb.train(self.xgb_param, dmatrix, self.train_round)
        return self.booster

    def predict(self, samples):
        """
        Predict

        Args:
            samples(list|numpy): an array of numpy array representing a batch
                of input samples.
        Returns:
            np.array representing labels
        """
        if isinstance(samples, list):
            samples = np.concatenate(samples, axis=0)
        dmatrix = xgb.DMatrix(data=samples, label=None)
        pred = self.booster.predict(dmatrix)
        return pred

    def save(self, path):
        """
        Save the trained XgbCostModel

        Args:
            path(str): path to save
        """
        assert (
            self.booster is not None
        ), "Calling save on a XgbCostModel not been trained"
        self.booster.save_model(path)

    def load(self, path):
        """
        Load the trained XgbCostModel

        Args:
            path(str): path to load
        """
        if self.booster is None:
            self.booster = xgb.Booster()
        self.booster.load_model(path)
        # Should we save/load config parameters? Not now because it is pre-set.
        # But we should do that here if that's changeable in the future.

    def update(self, samples, labels):
        # xgb doesn't support incremental training, we leave this method as TODO
        pass
