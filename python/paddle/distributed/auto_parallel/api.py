import numpy as np

import paddle
from paddle.io import BatchSampler, DataLoader, Dataset



class RandomDataset(Dataset):
    def __init__(self, seq_len, hidden, num_samples=100):
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            "float32"
        )
        label = np.random.uniform(size=[self.seq_len, self.hidden]).astype(
            'float32'
        )
        return (input, label)

    def __len__(self):
        return self.num_samples


class MlpModel(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[1024, 4096])
        self.w1 = self.create_parameter(shape=[4096, 1024])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


with paddle.LazyGuard():
    model = MlpModel()
for p in model.parameters():
    p.initialize()

dataset = RandomDataset(128, 1024)
sampler = BatchSampler(
    dataset,
    batch_size=4,
)
dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
)
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
loss_fn = paddle.nn.MSELoss()

for step, (inputs, labels) in enumerate(dataloader):
    logits = model(inputs)
    loss = loss_fn(logits, labels)
    loss.backward()
    opt.step()
    opt.clear_grad()
