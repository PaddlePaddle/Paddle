import paddle
from paddle.incubate.nn.functional import moe_gate_dispatch_permute

# 定义输入参数
num_rows = 10  # 示例行数
hidden_size = 128  # 隐藏层维度
num_experts = 4  # 专家数
world_size = 2  # 分布式世界大小
k = 2  # 选择的Top-k专家
capacity = 5  # 每个专家的处理容量

# 确保num_experts可以被world_size整除
assert num_experts % world_size == 0

# 生成输入数据
x = paddle.randn([num_rows, hidden_size], dtype='float32')
gate_logits = paddle.randn([num_rows, num_experts], dtype='float32')

# 可选的修正偏差
corr_bias = paddle.randn([num_rows], dtype='float32')

# 调用封装的API
y, combine_weights, scatter_index, expert_offset, expert_id = moe_gate_dispatch_permute(
    x=x, 
    gate_logits=gate_logits, 
    corr_bias=corr_bias, 
    k=k, 
    capacity=capacity, 
    world_size=world_size
)

# 打印输出结果的形状和类型，验证结果
print("Output y shape:", y.shape)
print("Combine weights shape:", combine_weights.shape)
print("Scatter index shape:", scatter_index.shape)
print("Expert offset shape:", expert_offset.shape)
print("Expert ID shape:", expert_id.shape)