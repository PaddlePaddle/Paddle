import unittest
import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.io import DataLoader, Dataset

from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    SharedLayerDesc,
    PipelineLayer,
)

batch_size = 5
micro_batch_size = 1

class RandomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        input_ids = np.random.random([5]).astype('int64')
        label = np.random.randint(0, 5, (5)).astype('int64')
        return input_ids, label

    def __len__(self):
        return self.num_samples


vocab_size = 1024
hidden_size = 64

class EmbeddingPipe(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed_tokens = nn.Embedding(kwargs["num_embeddings"], kwargs["embedding_dim"])

    def forward(self, input_ids):
        hidden_states = self.embed_tokens.forward(input_ids)
        return (hidden_states, input_ids)

    @property
    def embedding_weight(self):
        return getattr(self.embed_tokens, "weight")


class MTPEmbeddingPipe(EmbeddingPipe):
    def forward(self, args):
        hidden_states = args[0]
        input_ids = args[1]
        embed = super().forward(input_ids)
        output = embed[0] + hidden_states
        return (output, input_ids)


class LinearPipe(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_attr = None,
        bias_attr = None,
        name = None,
        layer_idx = 0
    ):
        self.layer_idx = layer_idx
        super().__init__(in_features, out_features, bias_attr=bias_attr)

    def forward(self, args):
        hidden_states = args[0]
        input_ids = args[1]
        output = super().forward(hidden_states)
        return (output, input_ids)


class CrossEntropyLossPipe(nn.loss.CrossEntropyLoss):
    def forward(self, logits, label):
        if isinstance(logits, tuple):
            logits = logits[0]
        return super().forward(logits, label)


class UnifiedPPModel(PipelineLayer):
    def __init__ (self, **kwargs):
        self._sequential_layers = []
        self.num_layer = 4

        #self.add_sequential_layer(
        #   SharedLayerDesc(
        #       key="embed_weight_share",
        #       layer_func=EmbeddingPipe,
        #       shared_weight_attr="embedding_weight",
        #       num_embeddings=vocab_size,
        #       embedding_dim=hidden_size,
        #   ),
        #   "embed",
        #)
        self.add_sequential_layer(
            LayerDesc(
                EmbeddingPipe,
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
            ), "embed"
        )

        for i in range(self.num_layer):
            self.add_sequential_layer(
                LayerDesc(
                    LinearPipe,
                    hidden_size,
                    hidden_size,
                    bias_attr=False,
                    layer_idx=i,
                ), f"layer.{i}"
            )

        self.add_sequential_layer(
           SharedLayerDesc(
               key="embed_weight_share",
               layer_func=MTPEmbeddingPipe,
               shared_weight_attr="embedding_weight",
               num_embeddings=vocab_size,
               embedding_dim=hidden_size,
           ),
           "embed_shared",
        )

        self.add_sequential_layer(
           LayerDesc(
               LinearPipe,
               hidden_size,
               hidden_size,
               bias_attr=False,
               layer_idx=self.num_layer
           ), "last_layer"
        )

        super().__init__(layers=self.get_sequential_layer(), loss_fn=CrossEntropyLossPipe(), **kwargs)

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        self._sequential_layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layer(self):
        return [x["layer"] for x in self._sequential_layers]



class TestDistPPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
        return scheduler, optimizer

    def wrapper_mix_precision(self, model, optimizer):
        return model, optimizer

    def test_unified_pp_model(self):
        unified_model_pp = UnifiedPPModel(num_stages=self.pipeline_parallel_size)
        unified_scheduler_pp, unified_optimizer_pp = self.build_optimizer(unified_model_pp)
        unified_model_pp, unified_optimizer_pp = self.wrapper_mix_precision(unified_model_pp, unified_optimizer_pp)
        unified_model_pp = fleet.distributed_model(unified_model_pp)
        unified_optimizer_pp = fleet.distributed_optimizer(unified_optimizer_pp)

        unified_model_nonpp = UnifiedPPModel(num_stages=1)
        unified_scheduler_nonpp, unified_optimizer_nonpp = self.build_optimizer(unified_model_nonpp)

        dataset = RandomDataset(5 * batch_size)

        train_reader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        for _, (input_ids, label) in enumerate(train_reader()):
            print(f"liyurui, input_ids is {input_ids.shape}, {input_ids.dtype}, label is {label.shape}, {label.dtype}")
            pp_loss = unified_model_pp.train_batch([input_ids, label], unified_optimizer_pp, unified_scheduler_pp)
            print(f"liyurui, pp_loss is {pp_loss}")

            nonpp_output = unified_model_nonpp(input_ids)
            loss_fn = nn.loss.CrossEntropyLoss()
            nonpp_loss = loss_fn(nonpp_output[0], label)
            print(f"liyurui, nonpp_loss is {nonpp_loss}")

            return


if __name__ == "__main__":
    unittest.main()