#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import layers

__all__ = [
    "simple_img_conv_pool",
    "sequence_conv_pool",
    "glu",
    "scaled_dot_product_attention",
    "SequenceDecoder",
]


def simple_img_conv_pool(input,
                         num_filters,
                         filter_size,
                         pool_size,
                         pool_stride,
                         act,
                         param_attr=None,
                         pool_type='max',
                         use_cudnn=True):
    conv_out = layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act,
        use_cudnn=use_cudnn)

    pool_out = layers.pool2d(
        input=conv_out,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn)
    return pool_out


def img_conv_group(input,
                   conv_num_filter,
                   pool_size,
                   conv_padding=1,
                   conv_filter_size=3,
                   conv_act=None,
                   param_attr=None,
                   conv_with_batchnorm=False,
                   conv_batchnorm_drop_rate=0.0,
                   pool_stride=1,
                   pool_type=None,
                   use_cudnn=True):
    """
    Image Convolution Group, Used for vgg net.
    """
    tmp = input
    assert isinstance(conv_num_filter, list) or \
        isinstance(conv_num_filter, tuple)

    def __extend_list__(obj):
        if not hasattr(obj, '__len__'):
            return [obj] * len(conv_num_filter)
        else:
            return obj

    conv_padding = __extend_list__(conv_padding)
    conv_filter_size = __extend_list__(conv_filter_size)
    param_attr = __extend_list__(param_attr)
    conv_with_batchnorm = __extend_list__(conv_with_batchnorm)
    conv_batchnorm_drop_rate = __extend_list__(conv_batchnorm_drop_rate)

    for i in xrange(len(conv_num_filter)):
        local_conv_act = conv_act
        if conv_with_batchnorm[i]:
            local_conv_act = None

        tmp = layers.conv2d(
            input=tmp,
            num_filters=conv_num_filter[i],
            filter_size=conv_filter_size[i],
            padding=conv_padding[i],
            param_attr=param_attr[i],
            act=local_conv_act,
            use_cudnn=use_cudnn)

        if conv_with_batchnorm[i]:
            tmp = layers.batch_norm(input=tmp, act=conv_act)
            drop_rate = conv_batchnorm_drop_rate[i]
            if abs(drop_rate) > 1e-5:
                tmp = layers.dropout(x=tmp, dropout_prob=drop_rate)

    pool_out = layers.pool2d(
        input=tmp,
        pool_size=pool_size,
        pool_type=pool_type,
        pool_stride=pool_stride,
        use_cudnn=use_cudnn)
    return pool_out


def sequence_conv_pool(input,
                       num_filters,
                       filter_size,
                       param_attr=None,
                       act="sigmoid",
                       pool_type="max"):
    conv_out = layers.sequence_conv(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        param_attr=param_attr,
        act=act)

    pool_out = layers.sequence_pool(input=conv_out, pool_type=pool_type)
    return pool_out


def glu(input, dim=-1):
    """
    The gated linear unit composed by split, sigmoid activation and elementwise
    multiplication. Specifically, Split the input into two equal sized parts
    :math:`a` and :math:`b` along the given dimension and then compute as
    following:

        .. math::

            {GLU}(a, b)= a \otimes \sigma(b)

    Refer to `Language Modeling with Gated Convolutional Networks
    <https://arxiv.org/pdf/1612.08083.pdf>`_.

    Args:
        input (Variable): The input variable which is a Tensor or LoDTensor.
        dim (int): The dimension along which to split. If :math:`dim < 0`, the
            dimension to split along is :math:`rank(input) + dim`.

    Returns:
        Variable: The Tensor variable with half the size of input.

    Examples:
        .. code-block:: python

            # x is a Tensor variable with shape [3, 6, 9]
            fluid.nets.glu(input=x, dim=1)  # shape of output: [3, 3, 9]
    """

    a, b = layers.split(input, num_or_sections=2, dim=dim)
    act_b = layers.sigmoid(x=b)
    out = layers.elementwise_mul(x=a, y=act_b)
    return out


def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 num_heads=1,
                                 dropout_rate=0.):
    """
    The dot-product attention.

    Attention mechanism can be seen as mapping a query and a set of key-value
    pairs to an output. The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed by a compatibility
    function (dot-product here) of the query with the corresponding key.

    The dot-product attention can be implemented through (batch) matrix
    multipication as follows:

        .. math::

            Attention(Q, K, V)= softmax(QK^\mathrm{T})V

    Refer to `Attention Is All You Need
    <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:

        queries (Variable): The input variable which should be a 3-D Tensor.
        keys (Variable): The input variable which should be a 3-D Tensor.
        values (Variable): The input variable which should be a 3-D Tensor.
        num_heads (int): Head number to compute the scaled dot product
                         attention. Default value is 1.
        dropout_rate (float): The dropout rate to drop the attention weight.
                              Default value is 0.

    Returns:

        Variable: A 3-D Tensor computed by multi-head scaled dot product
                  attention.

    Raises:

        ValueError: If input queries, keys, values are not 3-D Tensors.

    NOTE:
        1. When num_heads > 1, three linear projections are learned respectively
        to map input queries, keys and values into queries', keys' and values'.
        queries', keys' and values' have the same shapes with queries, keys
        and values.

        1. When num_heads == 1, scaled_dot_product_attention has no learnable
        parameters.

    Examples:
        .. code-block:: python

            # Suppose q, k, v are Tensors with the following shape:
            # q: [3, 5, 9], k: [3, 6, 9], v: [3, 6, 10]

            contexts = fluid.nets.scaled_dot_product_attention(q, k, v)
            contexts.shape  # [3, 5, 10]
    """
    if not (len(queries.shape) == len(keys.shape) == len(values.shape) == 3):
        raise ValueError(
            "Inputs quries, keys and values should all be 3-D tensors.")

    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(
            "The hidden size of queries and keys should be the same.")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(
            "The max sequence length in query batch and in key batch "
            "should be the same.")
    if keys.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of keys (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (keys.shape[-1], num_heads))
    if values.shape[-1] % num_heads != 0:
        raise ValueError("The hidden size of values (%d) must be divisible "
                         "by the number of attention heads (%d)." %
                         (values.shape[-1], num_heads))

    def __compute_qkv(queries, keys, values, num_heads):
        """
        Add linear projection to queries, keys, and values.

        Args:
            queries(Tensor): a 3-D input Tensor.
            keys(Tensor): a 3-D input Tensor.
            values(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads. Linearly project the inputs
                            ONLY when num_heads > 1.

        Returns:
            Tensor: linearly projected output Tensors: queries', keys' and
                    values'. They have the same shapes with queries, keys and
                    values.
        """

        if num_heads == 1:
            return queries, keys, values

        q = layers.fc(input=queries, size=queries.shape[-1], num_flatten_dims=2)
        k = layers.fc(input=keys, size=keys.shape[-1], num_flatten_dims=2)
        v = layers.fc(input=values, size=values.shape[-1], num_flatten_dims=2)
        return q, k, v

    def __split_heads(x, num_heads):
        """
        Reshape the last dimension of inpunt tensor x so that it becomes two
        dimensions.

        Args:
            x(Tensor): a 3-D input Tensor.
            num_heads(int): The number of heads.

        Returns:
            Tensor: a Tensor with shape [..., n, m/num_heads], where m is size
                    of the last dimension of x.
        """
        if num_heads == 1:
            return x

        hidden_size = x.shape[-1]
        # reshape the 3-D input: [batch_size, max_sequence_length, hidden_dim]
        # into a 4-D output:
        # [batch_size, max_sequence_length, num_heads, hidden_size_per_head].
        reshaped = layers.reshape(
            x=x,
            shape=list(x.shape[:-1]) + [num_heads, hidden_size // num_heads])

        # permuate the dimensions into:
        # [batch_size, num_heads, max_sequence_len, hidden_size_per_head]
        return layers.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def __combine_heads(x):
        """
        Reshape the last two dimensions of inpunt tensor x so that it becomes
        one dimension.

        Args:
            x(Tensor): a 4-D input Tensor with shape
                       [bs, num_heads, max_sequence_length, hidden_dim].

        Returns:
            Tensor: a Tensor with shape
                    [bs, max_sequence_length, num_heads * hidden_dim].
        """

        if len(x.shape) == 3: return x
        if len(x.shape) != 4:
            raise ValueError("Input(x) should be a 4-D Tensor.")

        trans_x = layers.transpose(x, perm=[0, 2, 1, 3])
        return layers.reshape(
            x=trans_x,
            shape=map(int, [
                trans_x.shape[0], trans_x.shape[1],
                trans_x.shape[2] * trans_x.shape[3]
            ]))

    q, k, v = __compute_qkv(queries, keys, values, num_heads)

    q = __split_heads(q, num_heads)
    k = __split_heads(k, num_heads)
    v = __split_heads(v, num_heads)

    key_dim_per_head = keys.shape[-1] // num_heads
    scaled_q = layers.scale(x=q, scale=key_dim_per_head**-0.5)
    product = layers.matmul(x=k, y=scaled_q, transpose_y=True)

    weights = layers.reshape(
        x=layers.reshape(
            x=product, shape=[-1, product.shape[-1]], act="softmax"),
        shape=product.shape)
    if dropout_rate:
        weights = layers.dropout(x, dropout_prob=dropout_rate, is_test=False)
    ctx_multiheads = layers.matmul(weights, v)
    return __combine_heads(ctx_multiheads)


pd = layers


class SequenceDecoder:
    '''
    SequenceDecoder is a helper class for sequence decoding tasks. It can be used to
    train a decoder and decode a sequence.
    '''

    class Cell:
        '''
        Cell is basic variable that is used to customize a decoder's logic. Every cell
        has two mode, the train mode and decode mode, and two kinds, step_input and state.
        Each `kind` has different datatypes in different mode, for example, a state will
        be an `rnn.memory` in train mode, but an `TensorArray` in decode mode.

        The Cell concept is introducted to make the logics that shared between different
        reusable. The states and item_id(word id) varialbe-like symbol should be reused,
        plus, the state update logic should be reused too. Each cell and the temporay
        variables that needed to be used across the logic can be added to Cell's static
        member `dic`, so all these symbols can be obtained by their name, that make it
        possible to write the same logic in different modes.
        '''
        modes = ('train', 'decode')
        kinds = ('step_input', 'state', 'other_shared')
        dic = {
            'train': {
                'step_input': {},
                'state': {},
                'others': {},
            },
            'decode': {
                'step_input': {},
                'state': {},
                'others': {},
            }
        }

        def __init__(self, kind, id, init_var, dtype='float32'):
            '''
            kind: state_input, state or other_shared
            id: identification of this cell.
            init_var: variable to initialize this cell.
                init_var can be an Variable or a dic like {'train': var0, 'decode': var1}
                so that different initialization can be porformed in different mode, this is
                important when defining states.
            seqdec: instance of SequenceDecoder.
            rnn: Dynamic rnn instance.
            '''
            self.kind = kind
            self.id = id
            self.init_var = init_var
            self.dtype = dtype
            self.rnn = None
            self.mode = None
            self.zero = pd.zeros(shape=[1], dtype='int64')

            assert kind in SequenceDecoder.Cell.kinds

        def set_mode(self, mode, seqdec):
            '''
            mode: str, either train or decode
            seqdec: instance of SequenceDecoder
            '''
            self.mode = mode
            self.seqdec = seqdec
            assert self.mode in SequenceDecoder.Cell.modes, \
            "invalid mode: {}, only {} are valid.".format(
                self.mode, SequenceDecoder.Cell.modes,)
            if self.mode == 'train':
                self.rnn = seqdec.train_rnn

        def create(self):
            assert self.mode is not None, "mode should be set first by calling `set_mode`"
            if self.mode == 'train':
                return self._create_train()
            else:
                return self._create_decode()

        def set_updater(self, updater):
            '''
            set update handler for state, the `updater` will be called like `updater(seq_decoder)`,
            the `seq_decoder` is an instance of `SequenceDecoder`.
            '''
            assert self.kind == 'state', 'updater is only needed in train mode'
            self.updater = updater

        def update(self):
            return self.updater(self.seqdec)

        @staticmethod
        def get(mode, kind, id, counter=None):
            item = SequenceDecoder.Cell.dic[mode][kind][id]
            if counter is None:
                return item
            return pd.array_read(array=item, i=counter)

        @staticmethod
        def add_temp_var(mode, id, var):
            dic = SequenceDecoder.Cell.dic
            assert id not in dic[mode][
                'others'], 'already a temporary var called %s there, change to another name' % id
            dic[mode]['others'][id] = var

        @staticmethod
        def get_temp_var(mode, id):
            return SequenceDecoder.Cell.dic[mode]['others'][id]

        def _create_train(self):
            dic = SequenceDecoder.Cell.dic
            init_var = self.init_var
            if type(self.init_var) is dict:
                init_var = init_var['train']
            if self.kind == 'step_input':
                print 'create var ', 'train', 'step_input', self.id
                self.input_array = None
                self.input = self.rnn.step_input(init_var)
                dic['train']['step_input'][self.id] = self.input
                return self.input
            else:
                print 'create var ', 'train', 'state', self.id
                self.state_array = None
                self.state = self.rnn.memory(init=self.init_var)
                dic['train']['state'][self.id] = self.state
                return self.state

        def _create_decode(self):
            init_var = self.init_var
            if type(self.init_var) is dict:
                init_var = init_var['decode']
            dic = SequenceDecoder.Cell.dic
            if self.kind == 'step_input':
                print 'create var ', 'decode', 'step_input', self.id
                self.input = None
                self.input_array = pd.create_array(self.dtype)
                dic['decode']['step_input'][id] = self.input_array
                pd.array_write(init_var, array=self.input_array, i=self.zero)
                return self.input_array
            else:
                print 'create var ', 'decode', 'state', self.id
                self.state = None
                self.state_array = pd.create_array(self.dtype)
                pd.array_write(init_var, array=self.state_array, i=self.zero)
                dic['decode']['state'][self.id] = self.state_array
                return self.state_array

    class InputCell(Cell):
        def __init__(self, id, init_var, dtype='float32'):
            SequenceDecoder.Cell.__init__(self, 'step_input', id, init_var,
                                          dtype)

        @staticmethod
        def get(mode, id):
            return SequenceDecoder.Cell.get(mode, 'step_input', id)

    class StateCell(Cell):
        def __init__(self, id, init_var, dtype='float32'):
            SequenceDecoder.Cell.__init__(self, 'state', id, init_var, dtype)

        @staticmethod
        def get(mode, id, counter=None):
            return SequenceDecoder.Cell.get(mode, 'state', id, counter)

    def __init__(self, item_id, states, scorer, other_step_inputs=[]):
        '''
        item_id: StepInput
        item_score: StepInput
        '''
        self.item_id = item_id
        self.states = states
        self.scorer = scorer
        self.other_step_inputs = other_step_inputs

        self.counter = pd.zeros(shape=[1], dtype='int64')
        # dynamic rnn is used in train mode.
        self.train_rnn = pd.DynamicRNN()

        self.mode = None

    def train(self):
        '''
        step_inputs: inputs that need to partition for each time step.
        static_inputs: similar to `step_inputs` but do not need backward.
        states_updates: state and updater.
        scorer: score calculator callback.
                will be called like `scorer(self)`
        '''
        self.mode = 'train'
        with self.train_rnn.block():
            # create item_id
            self.item_id.set_mode('train', self)
            self.item_id.create()
            print 'item_id', SequenceDecoder.Cell.dic
            # create step_inputs and states
            for x in self.other_step_inputs:
                x.set_mode('train', self)
                x.create()
            for state in self.states:
                state.set_mode('train', self)
                state.create()
            new_states = []
            for state in self.states:
                new_state = state.update()
                new_states.append(new_state)

            score = self.scorer(self)
            for no, state in enumerate(self.states):
                self.train_rnn.update_memory(state.state, new_states[no])
            self.train_rnn.output(score)
        return self.train_rnn()

    def decode(self, beam_size, max_length, topk, end_id):
        '''
        The decode phase of a decoder.

        The decode phase just shared a little with train phase, so it is
        a single method, and it do not need to split step input and rearrange
        the state, so it do not reuse the dynamic rnn.
        And it's more intuitive to write all the process of decoding, user can
        use this default decode, or just write their own logic according to the
        `decode` logic without the need to read a lot of framework codes.

        max_length: the max length the decoder can decode.
        '''
        self.mode = 'decode'
        # update mode
        self.item_id.set_mode('decode', self)
        self.item_id.create()  # create a tensor array
        for x in self.states:
            x.set_mode('decode', self)
        # step input are not needed in generation.

        array_len = pd.fill_constant(shape=[1], dtype='int64', value=max_length)

        for state in self.states:
            state.create()

        ids_array = self.item_id.input_array
        scores_array = pd.create_array('float32')

        # TODO(Superjomn) make initialization more convenient.
        init_ids = pd.data(
            name="init_ids", shape=[1], dtype="int64", lod_level=2)
        init_scores = pd.data(
            name="init_scores", shape=[1], dtype="float32", lod_level=2)

        pd.array_write(init_ids, array=ids_array, i=self.counter)
        pd.array_write(init_scores, array=scores_array, i=self.counter)

        # TODO there is an empty op that can prune the generation of the early stopped prefix.
        cond = pd.less_than(x=self.counter, y=array_len)

        while_op = pd.While(cond=cond)
        with while_op.block():
            pre_ids = pd.array_read(ids_array, i=self.counter)
            # use rnn cell to update rnn
            updated_states = []
            for state in self.states:
                current_state = state.update()
                updated_states.append(current_state)

            current_score = self.scorer()

            topk_scores, topk_indices = pd.topk(current_score, k=topk)
            selected_ids, selected_scores = pd.beam_search(
                pre_ids,
                topk_indices,
                topk_scores,
                beam_size,
                end_id=end_id,
                level=0)

            pd.increment(x=self.counter, value=1, in_place=True)
            # update the memories
            pd.array_write(selected_ids, array=ids_array, i=self.counter)
            pd.array_write(selected_scores, array=scores_array, i=self.counter)

            for no, state in enumerate(self.states):
                current_state = updated_states[no]
                pd.array_write(current_state, array=state.state, i=self.counter)

            pd.less_than(x=self.counter, y=array_len, cond=cond)

        translation_ids, translation_scores = pd.beam_search_decode(
            ids=ids_array, scores=scores_array)
        return translation_ids, translation_scores
