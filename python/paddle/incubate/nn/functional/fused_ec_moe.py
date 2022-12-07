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

from paddle.fluid.layer_helper import LayerHelper


def fused_ec_moe(
    x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, act):
    """
    Applies fused ec_moe kernel.

    This method requires SM_ARCH in sm75, sm80, sm86.

    Args:
        x (Tensor): the input Tensor. Its shape is [bsz, seq_len, d_model].
        gate (Tensor): the gate Tensor to choose expert. Its shape is [bsz, seq_len, e].
        bmm0_weight (Tensor): the first batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm0_bias (Tensor): the first batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        bmm1_weight (Tensor): the second batch matrix matmul weight. Its shape is [e, d_model, d_feed_forward].
        bmm1_bias (Tensor): the second batch matrix matmul bias. Its shape is [e, 1, d_feed_forward].
        act (int): the Activation Type. 0 means gelu, 1 means relu. 

    Returns:
        Tensor: the output Tensor.

    Examples:
        .. code-block:: python

            # required: gpu
            import paddle
            from paddle.incubate.nn.functional import fused_ec_moe

            batch = 10
            seq_len = 128
            d_model = 1024
            d_feed_forward = d_model * 4
            num_expert = 8
            x = paddle.randn([batch, seq_len, d_model])
            gate = paddle.randn([batch, seq_len, num_expert])
            bmm0_weight = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm0_bias = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm1_weight = paddle.randn([num_expert, d_model, d_feed_forward])
            bmm1_bias = paddle.randn([num_expert, d_model, d_feed_forward])
            out = fused_ec_moe(x, gate, bmm0_weight, bmm0_bias, bmm1_weight, bmm1_bias, act=0)
            print(out.shape) # [batch, seq_len, num_expert]

            # The naive EcMoE process is: 

            def expert_choice_gating(logits, capacity,batch_idx,expert_idx):
                gates = F.softmax(logits, -1)  # [batch, seq_len, num_expert]
                indices1_s = paddle.topk(logits.transpose([0, 2, 1]), k=capacity, axis=-1)[1].cast("int32")  # [batch, num_expert, capacity]
                seqlen_idx = indices1_s.reshape([-1]) # [batch * num_expert * capacity]
                gather_idx = paddle.stack([batch_idx, seqlen_idx, expert_idx], -1) # [batch * num_expert * capacity, 3]
                prob = paddle.gather_nd(gates, gather_idx)  # [batch * num_expert * seq_len]
                return prob, expert_idx, gather_idx, capacity

            self.gate = nn.Linear(d_model, num_expert, bias_attr=False)
            self.wi_w = self.create_parameter(shape=[num_expert, d_model, d_feed_forward], is_bias=False)
            self.wi_b = self.create_parameter(shape=[num_expert, 1, d_feed_forward], is_bias=True)
            self.act = nn.GeLU()
            self.wo_w = self.create_parameter(shape=[num_expert, d_feed_forward, d_model], is_bias=False)
            self.wo_b = self.create_parameter(shape=[num_expert, 1, d_model], is_bias=True)

            gate_logits = self.gate(x)  # [batch, seq_len, num_expert]
            # batch_idx, expert_idx = [batch * num_expert * seq_len]
            expert_prob_flatten, expert_idx_flatten, gather_idx, cap = expert_choice_gating(
                gate_logits, k, batch_idx, expert_idx)
            outputs = paddle.zeros_like(src)
            batch_prob = expert_prob_flatten.reshape([batch, num_expert, -1, 1]) #[batch, num_expert, capacity, 1]
            
            batch_idx = gather_idx[:,:2] #[batch * num_expert * capacity, 2]
            selected_token = src.gather_nd(batch_idx)  # [batch * num_expert * capacity, d_model]

            batch_selected_token = selected_token.reshape([batch, num_expert, -1, d_model]) # [batch, num_expert, capacity, d_model]
            batch_selected_token = batch_selected_token.transpose([1,0,2,3]).reshape([num_expert, -1, src.shape[-1]]) # [num_expert, batch * capacity, d_model]

            output = P.bmm(batch_selected_token, self.wi_w) + self.wi_b # [num_expert, batch * capacity, d_feed_forward]
            output = self.act(output)
            output = P.bmm(output, self.wo_w) + self.wo_b [num_expert, batch * capacity, d_model]

            output = output.transpose([1,0,2]).reshape([batch, -1, self.moe, src.shape[-1]]) # [batch, capacity, num_expert, d_model] 
            output = output.transpose([0,2,1,3]) # [batch, num_expert, capacity, d_model] 
            output = batch_prob * output # [batch, num_expert, capacity, d_model] 
            output = output.reshape([-1, src.shape[-1]]) # [batch * num_expert * capacity, d_model] 
            
            outputs = outputs.scatter_nd_add(batch_idx, output)

    """
    helper = LayerHelper('fused_moe', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type='moe',
        inputs={'X': x, 
                'Gate': gate, 
                'Bmm0': bmm0_weight, 
                'Bias0': bmm0_bias,
                'Bmm1': bmm1_weight, 
                'Bias1': bmm1_bias, 
                },
        outputs={'OUT': out},
        attrs={'act': act},
    )
    return out
