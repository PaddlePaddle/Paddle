import paddle, paddle.nn as nn
from functools import partial
from paddle.incubate.nn.memory_efficient_attention import (
    memory_efficient_attention,
)
from paddle.nn.functional.flash_attention import flash_attention

class Attention(nn.Layer):
    def __init__(self, hidden_size=768, 
                 num_attention_heads=12, 
                 attention_probs_dropout_prob=0., 
                 attention_op="cutlass"):
        super().__init__()
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)

        self.attention_op = attention_op
        self.scale = self.attention_head_size**-0.5
    
    def head_to_batch_dim(self, tensor, transpose=True):
        tensor = tensor.reshape([0, 0, self.num_attention_heads, self.attention_head_size])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        return tensor

    def batch_to_head_dim(self, tensor, transpose=True):
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor
    
    def forward(self, hidden_states, context=None, attention_op=None):
        if context is None:
            context = hidden_states 
            
        attention_op = attention_op or self.attention_op
        transpose = False if attention_op in ['cutlass', 'flash'] else True
        if attention_op == "flash":
            self.to(dtype="float16")
            hidden_states = hidden_states.cast('float16')
            context = context.cast('float16')
        fn = partial(self.head_to_batch_dim, transpose=transpose)
        q = fn(self.query(hidden_states))
        k = fn(self.key(context))
        v = fn(self.value(context))
        
        if transpose:
            attention_scores = paddle.matmul(q, k, transpose_y=True) * self.scale
            attention_probs = paddle.nn.functional.softmax(attention_scores, axis=-1)
            hidden_states = paddle.matmul(attention_probs, v)
        else:
            if self.attention_op == "cutlass":
                print('x'*100)
                hidden_states = memory_efficient_attention(
                    q,
                    k,
                    v,
                    None,
                    p=0.0,
                    scale=self.scale,
                    training=True,
                )
            elif self.attention_op == "flash":
                hidden_states = flash_attention(
                    q,
                    k,
                    v,
                    dropout=0.0,
                    causal=False,
                    return_softmax=False,
                )[0]
                
        hidden_states = self.batch_to_head_dim(hidden_states, transpose=transpose)
        return hidden_states
    

attn = Attention()
def freeze_params(model):
    for param in model.parameters():
        param.stop_gradient = True
# 固定模型参数，我们不需要训练
freeze_params(attn)


hidden_states = paddle.randn((1, 16, 768))
context = paddle.randn((1, 16, 768))
context.stop_gradient = False
attention_op = "cutlass" # 或者 'flash'
o = attn(hidden_states=hidden_states, 
    context=context,
    attention_op=attention_op)
o.mean().backward()
o