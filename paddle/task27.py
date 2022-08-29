No.27：为 Paddle 新增 paddle.incubate.sparse.any 稀疏 API
技术标签：深度学习框架，Python，C++，CUDA

任务难度：基础

详细描述：针对 Paddle 的两种稀疏 Tensor 存储格式 COO 与 CSR，需要新增 any 的计算逻辑，一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

提交内容

API 的设计文档，并提 PR 至 community repo 的 rfcs/APIs 目录；
Python 实现代码 & 英文 API 文档，在 Paddle repo 的 python/paddle/incubate/sparse/unary.py 文件；
C++ kernel 实现代码，在Paddle repo 的paddle/phi/kernels/sparse/ 目录中；
单测代码，在 Paddle repo 新建 python/paddle/fluid/tests/unittests/test_sparse_any_op.py 文件；
yaml 文件，前反向分别添加到python/paddle/utils/code_gen/sparse_api.yaml、python/paddle/utils/code_gen/sparse_bw_api.yaml 文件中。
中文 API 文档，在 docs repo 的 docs/api/paddle/incubate/sparse 目录。
技术要求

熟悉稀疏 COO/CSR 存储格式，Paddle 的 SparseCooTensor/SparseCsrTensor 数据结构；
熟悉稀疏 Tensor的 any 在 COO/CSR 存储格式下的计算逻辑；
熟练掌握 Python、C++、CUDA 代码编写。