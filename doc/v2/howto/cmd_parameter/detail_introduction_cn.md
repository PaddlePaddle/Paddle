# 细节描述

## 通用

* `--job`
  - 工作模式，包括: **train, test, checkgrad**，其中checkgrad主要为开发者使用，使用者不需要关心。
  - 类型: string (默认: train)

* `--config`
  - 用于指定网络配置文件。
  - 类型: string (默认: null).

* `--use_gpu`
  - 训练过程是否使用GPU，设置为true使用GPU模式，否则使用CPU模式。
  - 类型: bool (默认: 1).

* `--local`
  - 训练过程是否为本地模式，设置为true使用本地训练或者使用集群上的一个节点，否则使用多机训练。
  - 类型: bool (默认: 1).

* `--trainer_count`
  - 指定一台机器上使用的线程数。例如，trainer_count = 4, 意思是在GPU模式下使用4个GPU，或者在CPU模式下使用4个线程。每个线程（或GPU）分配到当前数据块样本数的四分之一。也就是说，如果在训练配置中设置batch_size为512，每个线程分配到128个样本用于训练。
  - 类型: int32 (默认: 1).

* `--num_passes`
  - 当模式为`--job=train`时, 该参数的意思是训练num_passes轮。每轮会将数据集中的所有训练样本使用一次。当模式为`--job=test`时，意思是使用第test_pass个模型到第 num_passes-1 个模型测试数据。
  - 类型: int32 (默认: 100).

* `--config_args`
  - 传递给配置文件的参数。格式: key1=value1,key2=value2.
  - 类型: string (默认: null).

* `--version`
  - 是否打印版本信息。
  - 类型: bool (默认: 0).

* `--show_layer_stat`
  - 是否显示**每个批次数据**中每层的数值统计.
  - 类型: bool (默认: 0).

## 训练

* `--log_period`
  - 每log_period个批次打印日志进度.
  - 类型: int32 (默认: 100).

* `--dot_period`
  - 每dot_period个批次输出符号'.'.
  - 类型: int32 (默认: 1).

* `--saving_period`
  - 每saving_period轮保存训练参数.
  - 类型: int32 (默认: 1).

* `--save_dir`
  - 保存模型参数的目录，需要明确指定，但不需要提前创建。
  - 类型: string (默认: null).

* `--start_pass`
  - 从start_pass轮开始训练，会加载上一轮的参数。
  - 类型: int32 (默认: 0).

* `--show_parameter_stats_period`
  - 在训练过程中每show_parameter_stats_period个批次输出参数统计。默认不显示。
  - 类型: int32 (默认: 0).

* `--save_only_one`
  - 只保存最后一轮的参数，而之前的参数将会被删除。
  - 类型: bool (默认: 0).

* `--load_missing_parameter_strategy`
  - 当模型参数不存在时，指定加载的方式。目前支持fail/rand/zero三种操作.
    - `fail`: 程序直接退出.
    - `rand`: 根据网络配置中的**initial\_strategy**采用均匀分布或者高斯分布初始化。均匀分布的范围是: **[mean - std, mean + std]**, 其中mean和std是训练配置中的参数.
    - `zero`: 所有参数置为零.
  - 类型: string (默认: fail).

* `--init_model_path`
   - 初始化模型的路径。如果设置该参数，start\_pass将不起作用。同样也可以在测试模式中指定模型路径。
   - 类型: string (默认: null).

* `--saving_period_by_batches`
   - 在一轮中每saving_period_by_batches个批次保存一次参数。
   - 类型: int32 (默认: 0).

* `--log_error_clipping`
  - 当在网络层配置中设置**error_clipping_threshold**时，该参数指示是否打印错误截断日志。如果为true，**每批次**的反向传播将会打印日志信息。该截断会影响**输出的梯度**.
  - 类型: bool (默认: 0).

* `--log_clipping`
  - 当在训练配置中设置**gradient_clipping_threshold**时，该参数指示是否打印日志截断信息。该截断会影响**权重更新的梯度**.
  - 类型: bool (默认: 0).

* `--use_old_updater`
  - 是否使用旧的RemoteParameterUpdater。 默认使用ConcurrentRemoteParameterUpdater，主要为开发者使用，使用者通常无需关心.
  - 类型: bool (默认: 0).

* `--enable_grad_share`
  - 启用梯度参数的阈值，在多CPU训练时共享该参数.
  - 类型: int32 (默认: 100 \* 1024 \* 1024).

* `--grad_share_block_num`
  - 梯度参数的分块数目，在多CPU训练时共享该参数.
  - 类型: int32 (默认: 64).

## 测试

* `--test_pass`
  - 加载test_pass轮的模型用于测试.
  - 类型: int32 (默认: -1).

* `--test_period`
   - 如果为0，每轮结束时对所有测试数据进行测试；如果不为0，每test_period个批次对所有测试数据进行测试.
  - 类型: int32 (默认: 0).

* `--test_wait`
  - 指示当指定轮的测试模型不存在时，是否需要等待该轮模型参数。如果在训练期间同时发起另外一个进程进行测试，可以使用该参数.
  - 类型: bool (默认: 0).

* `--model_list`
  - 测试时指定的存储模型列表的文件.
  - 类型: string (默认: "", null).

* `--predict_output_dir`
  - 保存网络层输出结果的目录。该参数在网络配置的Outputs()中指定，默认为null，意思是不保存结果。在测试阶段，如果你想要保存某些层的特征图，请指定该目录。需要注意的是，网络层的输出是经过激活函数之后的值.
  - 类型: string (默认: "", null).

* `--average_test_period`
  - 使用`average_test_period`个批次的参数平均值进行测试。该参数必须能被FLAGS_log_period整除，默认为0，意思是不使用平均参数执行测试.
  - 类型: int32 (默认: 0).

* `--distribute_test`
  - 在分布式环境中测试，将多台机器的测试结果合并.
  - 类型: bool (默认: 0).

* `--predict_file`
  - 保存预测结果的文件名。该参数默认为null，意思是不保存结果。目前该参数仅用于AucValidationLayer和PnpairValidationLayer层，每轮都会保存预测结果.
  - 类型: string (默认: "", null).

## GPU

* `--gpu_id`
  - 指示使用哪个GPU核.
  - 类型: int32 (默认: 0).

* `--allow_only_one_model_on_one_gpu`
  - 如果为true，一个GPU设备上不允许配置多个模型.
  - 类型: bool (默认: 1).

* `--parallel_nn`
  - 指示是否使用多线程来计算一个神经网络。如果为false，设置gpu_id指定使用哪个GPU核（训练配置中的设备属性将会无效）。如果为true，GPU核在训练配置中指定（gpu_id无效）.
  - 类型: bool (默认: 0).

* `--cudnn_dir`
  - 选择路径来动态加载NVIDIA CuDNN库，例如，/usr/local/cuda/lib64. [默认]: LD_LIBRARY_PATH
  - 类型: string (默认: "", null)

* `--cuda_dir`
  - 选择路径来动态加载NVIDIA CUDA库，例如，/usr/local/cuda/lib64. [默认]: LD_LIBRARY_PATH
  - 类型: string (默认: "", null)

* `--cudnn_conv_workspace_limit_in_mb`
  - 指定cuDNN的最大工作空间容限，单位是MB，默认为4096MB=4GB. 
  - 类型: int32 (默认: 4096MB=4GB)

## 自然语言处理(NLP): RNN/LSTM/GRU
* `--rnn_use_batch`
  - 指示在简单的RecurrentLayer层的计算中是否使用批处理方法.
  - 类型: bool (默认: 0).

* `--prev_batch_state`
  - 标识是否为连续的batch计算.
  - 类型: bool (默认: 0).

* `--beam_size`
  - 集束搜索使用广度优先搜索的方式构建查找树。在树的每一层上，都会产生当前层状态的所有继承结果，按启发式损失的大小递增排序。然而，每层上只能保存固定数目个最好的状态，该数目是提前定义好的，称之为集束大小.
  - 类型: int32 (默认: 1).

* `--diy_beam_search_prob_so`
  - 用户可以自定义beam search的方法，编译成动态库，供PaddlePaddle加载。 该参数用于指定动态库路径.
  - 类型: string (默认: "", null).

## 数据支持(DataProvider)

* `--memory_threshold_on_load_data`
  - 内存容限阈值，当超过该阈值时，停止加载数据.
  - 类型: double (默认: 1.0).

## 单元测试

* `--checkgrad_eps`
  - 使用checkgrad模式时的参数变化大小.
  - 类型: double (默认: 1e-05).

## 参数服务器和分布式通信

* `--start_pserver`
  - 指示是否开启参数服务器(parameter server).
  - 类型: bool (默认: 0).

* `--pservers`
  - 参数服务器的IP地址，以逗号间隔.
  - 类型: string (默认: "127.0.0.1").

* `--port`
  - 参数服务器的监听端口.
  - 类型: int32 (默认: 20134).

* `--ports_num`
  - 发送参数的端口号，根据默认端口号递增.
  - 类型: int32 (默认: 1).

* `--trainer_id`
  - 在分布式训练中，每个训练节点必须指定一个唯一的id号，从0到num_trainers-1。0号训练节点是主训练节点。使用者无需关心这个参数.
  - 类型: int32 (默认: 0).

* `--num_gradient_servers`
  - 梯度服务器的数量，该参数在集群提交环境中自动设置.
  - 类型: int32 (默认: 1).

* `--small_messages`
  - 如果消息数据太小，建议将该参数设为true，启动快速应答，无延迟.
  - 类型: bool (默认: 0).

* `--sock_send_buf_size`
  - 限制套接字发送缓冲区的大小。如果仔细设置的话，可以有效减小网络的阻塞.
  - 类型: int32 (默认: 1024 \* 1024 \* 40).

* `--sock_recv_buf_size`
  - 限制套接字接收缓冲区的大小.
  - 类型: int32 (默认: 1024 \* 1024 \* 40).

* `--parameter_block_size`
  - 参数服务器的参数分块大小。如果未设置，将会自动计算出一个合适的值.
  - 类型: int32 (默认: 0).

* `--parameter_block_size_for_sparse`
  - 参数服务器稀疏更新的参数分块大小。如果未设置，将会自动计算出一个合适的值.
  - 类型: int32 (默认: 0).

* `--log_period_server`
  - 在参数服务器终端每log_period_server个批次打印日志进度.
  - 类型: int32 (默认: 500).

* `--loadsave_parameters_in_pserver`
  - 在参数服务器上加载和保存参数，只有当设置了sparse_remote_update参数时才有效.
  - 类型: bool (默认: 0).

* `--pserver_num_threads`
  - 同步执行操作的线程数.
  - 类型: bool (默认: 1).

* `--ports_num_for_sparse`
  - 发送参数的端口号，根据默认值递增(port + ports_num)，用于稀疏训练中.
  - 类型: int32 (默认: 0).

* `--nics`
  - 参数服务器的网络设备名称，已经在集群提交环境中完成设置.
  - 类型: string (默认: "xgbe0,xgbe1").

* `--rdma_tcp`
  - 使用rdma还是tcp传输协议，该参数已经在集群提交环境中完成设置.
  - 类型: string (默认: "tcp").

## 异步随机梯度下降(Async SGD)
* `--async_count`
  - 定义异步训练的长度，如果为0，则使用同步训练.
  - 类型: int32 (默认: 0).

* `--async_lagged_ratio_min`
  - 控制`config_.async_lagged_grad_discard_ratio()`的最小值.
  - 类型: double (默认: 1.0).

* `--async_lagged_ratio_default`
  - 如果在网络配置中未设置async_lagged_grad_discard_ratio，则使用该参数作为默认值.
  - 类型: double (默认: 1.5).

## 性能调优(Performance Tuning)

* `--log_barrier_abstract`
  - 如果为true，则显示阻隔性能的摘要信息.
  - 类型: bool (默认: 1).

* `--log_barrier_show_log`
  - 如果为true，则总会显示阻隔摘要信息，即使间隔很小.
  - 类型: bool (默认: 0).

* `--log_barrier_lowest_nodes`
  - 最少显示多少个节点.
  - 类型: int32 (默认: 5).

* `--check_sparse_distribution_in_pserver`
  - 指示是否检查所有参数服务器上的稀疏参数的分布是均匀的.
  - 类型: bool (默认: 0).

* `--show_check_sparse_distribution_log`
  - 指示是否显示参数服务器上的稀疏参数分布的日志细节.
  - 类型: bool (默认: 0).

* `--check_sparse_distribution_batches`
  - 每运行多少个批次执行一次稀疏参数分布的检查.
  - 类型: int32 (默认: 100).

* `--check_sparse_distribution_ratio`
  - 如果检查到分配在不同参数服务器上的参数的分布不均匀次数大于check_sparse_distribution_ratio *  check_sparse_distribution_batches次，程序停止.
  - 类型: double (默认: 0.6).

* `--check_sparse_distribution_unbalance_degree`
  - 不同参数服务器上数据大小的最大值与最小值的比率.
  - 类型: double (默认: 2).

## 矩阵/向量/随机数
* `--enable_parallel_vector`
  - 启动并行向量的阈值.
  - 类型: int32 (默认: 0).

* `--seed`
  - 随机数的种子。srand(time)的为0.
  - 类型: int32 (默认: 1)

* `--thread_local_rand_use_global_seed`
  - 是否将全局种子应用于本地线程的随机数.
  - 类型: bool (默认: 0).
