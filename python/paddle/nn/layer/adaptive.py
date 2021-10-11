# -*- coding: utf-8 -*-

from collections import namedtuple

import paddle
from paddle import Tensor
from typing import List, Sequence

from ...fluid.dygraph import layers
from ...fluid.dygraph.container import LayerList, Sequential  #DEFINE_ALIAS
from .common import Linear
from ..functional import log_softmax, one_hot


_ASMoutput = namedtuple('_ASMoutput', ['output', 'loss'])


class AdaptiveLogSoftmaxWithLoss(layers.Layer):
    r"""Efficient softmax approximation as described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Cissé, David Grangier, and Hervé Jégou
    <https://arxiv.org/abs/1609.04309>`__.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})`
        - target: :math:`(N)` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """

    in_features: int
    n_classes: int
    cutoffs: List[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: LayerList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.,
        head_bias: bool = False,
    ) -> None:
        super(AdaptiveLogSoftmaxWithLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, self.head_size, bias_attr=self.head_bias)
        self.tail = LayerList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias_attr=False),
                Linear(hsz, osz, bias_attr=False),
            )

            self.tail.append(projection)

    def forward(self, input: Tensor, target: Tensor) -> _ASMoutput:
        if input.shape[0] != target.shape[0]:
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.shape[0]

        output = paddle.zeros([batch_size])
        gather_inds = paddle.empty([batch_size])

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx).logical_and(target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue
            if i == 0:

                # gather_inds.index_copy_(0, row_indices, target[target_mask])
                scatter_output = paddle.scatter_nd(row_indices.unsqueeze(1), target.masked_select(target_mask), gather_inds.shape)
                gather_inds = gather_inds * (scatter_output == 0) + scatter_output

            else:
                relative_target = target.masked_select(target_mask) - low_idx
                input_subset = input.index_select(row_indices, 0)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                # gather_inds.index_fill_(0, row_indices, cluster_index)
                scatter_output = paddle.scatter_nd(row_indices.unsqueeze(1), paddle.ones(row_indices.shape)*cluster_index, gather_inds.shape)
                gather_inds = (gather_inds * (scatter_output != cluster_index) + scatter_output).astype(paddle.int64)

                cluster_logprob = log_softmax(cluster_output, axis=1)
                # local_logprob = cluster_logprob.gather(relative_target.unsqueeze(1), 1)
                local_logprob = (one_hot(relative_target, cluster_logprob.shape[-1]) * cluster_logprob).sum(1).unsqueeze(1)
                # output.index_copy_(0, row_indices, local_logprob.squeeze(1))
                scatter_output = paddle.scatter_nd(row_indices.unsqueeze(1), local_logprob.squeeze(1), output.shape)
                output = output * (scatter_output == 0) + scatter_output

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = log_softmax(head_output, axis=1)
        # output += head_logprob.gather(gather_inds.unsqueeze(1), 1).squeeze()
        output += (paddle.nn.functional.one_hot(gather_inds, head_logprob.shape[1]) * head_logprob).sum(1)
        loss = (-output).mean()

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = paddle.empty((head_output.shape[0], self.n_classes))
        head_logprob = log_softmax(head_output, axis=1)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](input)
            cluster_logprob = log_softmax(cluster_output, axis=1)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r""" Computes log probabilities for all :math:`\texttt{n\_classes}`

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """

        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r""" This is equivalent to `self.log_pob(input).argmax(axis=1)`,
        but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """

        head_output = self.head(input)
        output = paddle.argmax(head_output, axis=1)
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return paddle.argmax(log_prob, axis=1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = paddle.argmax(log_prob, axis=1)
            return output
