# Paddle 基础评审规则

本文件是每次 Paddle 评审都必须加载的基础规则。根据变更范围应用相关条目，并在评论中给出路径、行号和可观察影响。

## 功能正确性与兼容性

- 核对 PR 描述与实现是否一致，包括输入输出、默认值、支持的 shape/dtype、设备范围和测试范围。
- 检查空输入、零值、负值、上下限、重复输入、非法格式、异常路径和资源失败，避免只覆盖正常流程。
- 公共 Python/C++ API、配置项、默认值、state dict、序列化格式或算子 schema 变化必须保持向后兼容，或提供明确迁移方案。
- 跨模块变更必须检查接口两端，确认参数顺序、返回值、dtype、shape、设备、place、stream 和生命周期约定一致。
- PIR、动态图和静态图同时支持时，检查三条路径的行为、自动生成代码和回退逻辑是否一致。

## 算子、Kernel 与设备

- 修改 `paddle/phi/ops/yaml/` 时同步检查 InferMeta、kernel 声明和实现、注册名、Python 封装、代码生成依赖及前向/反向配置。
- 检查 `paddle/phi/infermeta/` 的 shape/dtype 推导、输入校验、空 Tensor 和边界行为；InferMeta 不应假定 kernel 一定可用。
- 检查 `paddle/phi/kernels/` 的设备与 stream、整数宽度、索引边界、内存分配、kernel launch 错误和异步生命周期。
- CPU、CUDA、ROCm、XPU、Custom Device 或其他后端新增路径必须有明确能力保护，并与已有实现保持语义一致或提供可验证回退。
- 修改 `python/paddle/` API 时检查动态图、PIR、静态图、类型转换、梯度和文档示例；不能只验证 Python 表面调用成功。
- 关注隐式同步和多余拷贝，如 `.cpu()`、`.numpy()`、不必要的 `to_tensor()`、`contiguous()` 或 device/place 往返。

## 分布式训练与并行

- 修改 `python/paddle/distributed/`、`python/paddle/incubate/distributed/`、`paddle/phi/infermeta/spmd_rules/` 或通信逻辑时，检查进程组创建/销毁、rank 成员、重新初始化和错误路径。
- 集合通信必须使用正确的通信组，并保证各 rank 的调用次数、顺序、tensor shape、dtype 和参与条件对称；不能用单进程 mock 代替多进程证据。
- 检查 DP、TP、PP、SP、EP、CP、ZeRO 和自动并行组合下的局部 shape、切分、reshard、位置编码、随机数同步、共享权重和流水线边界。
- MoE 路径重点检查 top-k、归一化、token dispatch/combine 顺序、expert index、容量/丢弃策略、共享专家及梯度对称性。
- 重计算和 CUDA Graph 变更必须保持前向/反向一致、随机状态正确、地址稳定，并为动态 shape 或不支持设备保留安全回退路径。

## 数值、性能与资源

- 前向、反向和高阶梯度的 dtype、累加精度、NaN/Inf、溢出、归一化和容差必须与 API 契约一致；不能只比较单个正常样例。
- FP16、BF16、FP8、量化、融合算子、CINN、TensorRT 或 Triton 路径必须有明确的硬件/shape/dtype 保护，并与非融合实现比较结果或说明回退。
- 检查低效循环、重复计算、显存峰值、内存泄漏、句柄/事件/通信组未释放及异常路径中的资源清理；性能意见必须说明热点或复杂度影响。
- 修改 `cmake/`、`paddle/**/CMakeLists.txt`、代码生成脚本、依赖或 lockfile 时，确认所有目标、平台和构建配置仍能解析且没有无关依赖升级。

## 安全与错误处理

- 禁止硬编码密钥、令牌和凭据；外部输入必须经过校验，不能直接拼接到 shell、路径、SQL 或不安全反序列化操作。
- 检查权限、认证、临时文件和敏感数据处理是否符合最小权限原则，避免把调试信息或环境变量写入日志。
- 不使用 Python `assert` 承担运行时输入校验；应抛出明确异常并保留有效错误信息。
- 避免无处理的宽泛 `except Exception`，确认文件、线程、显存、stream、通信组和临时资源在失败路径中正确释放。

## 测试质量与验证

- 行为变化应放入最接近变更模块的现有目录，例如 `test/legacy_test/`、`test/cpp/`、`test/auto_parallel/`、`test/collective/`、`test/custom_kernel/`、`test/custom_runtime/`、`test/custom_op/` 或对应 `python/` 测试目录；分布式行为需要代表性的多进程覆盖。
- 算子测试应覆盖前向、反向、PIR/动态图（适用时）、shape/dtype/设备边界和异常路径；测试应验证核心结果而不是只断言不报错。
- 测试之间不得相互依赖；外部网络、文件系统、随机状态或服务应隔离，不能依赖不稳定的共享状态。
- 断言不能吞掉异常、使用 `assert True` 或 mock 掉被测函数来制造成功；浮点结果使用合理容差，重复场景优先参数化。
- 修改 `ci/rules/` 时同步检查 `ci/rule-tests/` 和快照；新增 blacklist 或跳过条件不能替代回归测试。
- 将 `git diff --check`、相关静态检查和可执行测试分开记录；Paddle、CUDA、分布式或特定硬件不可用时，不把未运行包装成通过。

## PR 信息与评审评论

- PR 标题遵循 Paddle 约定的 `[类别] 简要说明` 格式，描述至少说明动机、解决的问题、主要改动和验证方式。
- 检查 PR 描述中的精度变化、兼容性和设备范围声明是否与实现一致；发现不一致时指出具体代码证据。
- 评论必须具体、可执行并解释影响原因；优先报告会导致错误、回归、数据损坏、安全风险或无法构建的问题，不报告纯风格偏好。
