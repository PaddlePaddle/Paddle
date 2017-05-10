```eval_rst
..  _cmd_detail_introduction:
```

# Detail Description

## Common

* `--job`
  - Job mode, including: **train, test, checkgrad**, where checkgrad is mainly for developers and users do not need to care about.
  - type: string (default: train)

* `--config`
  - Use to specfiy network configure file.
  - type: string (default: null).

* `--use_gpu`
  - Whether to use GPU for training, false is cpu mode and true is gpu mode.
  - type: bool (default: 1).

* `--local`
  - Whether the training is in local mode or not. True when training locally or using one node in cluster. False when using multiple machines in cluster.
  - type: bool (default: 1).

* `--trainer_count`
  - Define the number of threads used in one machine. For example, trainer_count = 4, means use 4 GPU in GPU mode and 4 threads in CPU mode. Each thread (or GPU) is assigned to 1/4 samples in current batch. That is to say, if setting batch_size of 512 in trainer config, each thread train 128 samples.
  - type: int32 (default: 1).

* `--num_passes`
   - When `--job=train`, means training for num_passes passes. One pass means training all samples in dataset one time. When `--job=test`, means testing data from model of test_pass to  model of (num_passes - 1).
   - type: int32 (default: 100).

* `--config_args`
  - arguments passed to config file. Format: key1=value1,key2=value2.
  - type: string (default: null).

* `--version`
  - Whether to print version information.
  - type: bool (default: 0).

* `--show_layer_stat`
  - Whether to show the statistics of each layer **per batch**.
  - type: bool (default: 0).

## Train

* `--log_period`
  - Log progress every log_period batches.
  - type: int32 (default: 100).

* `--dot_period`
  - Print '.' every dot_period batches.
  - type: int32 (default: 1).

* `--saving_period`
  - Save parameters every saving_period passes
  - type: int32 (default: 1).

* `--save_dir`
  - Directory for saving model parameters. It needs to be specified, but no need to be created in advance.
  - type: string (default: null).

* `--start_pass`
  - Start training from this pass. It will load parameters from the previous pass.
  - type: int32 (default: 0).

* `--show_parameter_stats_period`
  - Show parameter statistic during training every show_parameter_stats_period batches. It will not show by default.
  - type: int32 (default: 0).

* `--save_only_one`
  - Save the parameters only in last pass, while the previous parameters will be removed.
  - type: bool (default: 0).

* `--load_missing_parameter_strategy`
  - Specify the loading operation when model file is missing. Now support fail/rand/zero three operations.
    - `fail`: program will exit.
    - `rand`: uniform or normal distribution according to **initial\_strategy** in network config. Uniform range is: **[mean - std, mean + std]**, where mean and std are configures in trainer config.
    - `zero`: all parameters are zero.
  - type: string (default: fail).

* `--init_model_path`
   - Path of the initialization model. If it was set, start\_pass will be ignored. It can be used to specify model path in testing mode as well.
   - type: string (default: null).

* `--saving_period_by_batches`
   - Save parameters every saving_period_by_batches batches in one pass.
   - type: int32 (default: 0).

* `--log_error_clipping`
  - Whether to print error clipping log when setting **error_clipping_threshold** in layer config. If it is true, log will be printed in backward propagation **per batch**. This clipping effects on **gradient of output**.
  - type: bool (default: 0).

* `--log_clipping`
  - Enable print log clipping or not when setting **gradient_clipping_threshold** in trainer config. This clipping effects on **gradient w.r.t. (with respect to) weight**.
  - type: bool (default: 0).

* `--use_old_updater`
  - Whether to use the old RemoteParameterUpdater. Default use ConcurrentRemoteParameterUpdater. It is mainly for deverlopers and users usually do not need to care about.
  - type: bool (default: 0).

* `--enable_grad_share`
  - threshold for enable gradient parameter, which is shared for batch multi-cpu training.
  - type: int32 (default: 100 \* 1024 \* 1024).

* `--grad_share_block_num`
  - block number of gradient parameter, which is shared for batch multi-cpu training.
  - type: int32 (default: 64).

## Test

* `--test_pass`
  - Load parameter from this pass to test.
  - type: int32 (default: -1).

* `--test_period`
   - if equal 0, do test on all test data at the end of each pass. While if equal non-zero, do test on all test data every test_period batches.
  - type: int32 (default: 0).

* `--test_wait`
  - Whether to wait for parameter per pass if not exist. It can be used when user launch another process to perfom testing during the training process.
  - type: bool (default: 0).

* `--model_list`
  - File that saves the model list when testing. 
  - type: string (default: "", null).

* `--predict_output_dir`
  - Directory that saves the layer output. It is configured in Outputs() in network config. Default, this argument is null, meaning save nothing. Specify this directory if you want to save feature map of some layers in testing mode. Note that, layer outputs are values after activation function.
  - type: string (default: "", null).

* `--average_test_period`
  - Do test on average parameter every `average_test_period` batches. It MUST be devided by FLAGS_log_period. Default 0 means do not test on average parameter.
  - type: int32 (default: 0).

* `--distribute_test`
  - Testing in distribute environment will merge results from multiple machines.
  - type: bool (default: 0).

* `--predict_file`
  - File name for saving predicted result. Default, this argument is null, meaning save nothing. Now, this argument is only used in AucValidationLayer and PnpairValidationLayer, and saves predicted result every pass.
  - type: string (default: "", null).

## GPU

* `--gpu_id`
  - Which gpu core to use.
  - type: int32 (default: 0).

* `--allow_only_one_model_on_one_gpu`
  - If true, do not allow multiple models on one GPU device.
  - type: bool (default: 1).

* `--parallel_nn`
  - Whether to use multi-thread to calculate one neural network or not. If false, use gpu_id specify which gpu core to use (the device property in trainer config will be ingored). If true, the gpu core is specified in trainer config (gpu_id will be ignored).
  - type: bool (default: 0).

* `--cudnn_dir`
  - Choose path to dynamic load NVIDIA CuDNN library, for instance, /usr/local/cuda/lib64. [Default]: LD_LIBRARY_PATH
  - type: string (default: "", null)

* `--cuda_dir`
  - Choose path to dynamic load NVIDIA CUDA library, for instance, /usr/local/cuda/lib64. [Default]: LD_LIBRARY_PATH
  - type: string (default: "", null)

* `--cudnn_conv_workspace_limit_in_mb`
  - Specify cuDNN max workspace limit, in units MB, 4096MB=4GB by default. 
  - type: int32 (default: 4096MB=4GB)

## NLP: RNN/LSTM/GRU
* `--rnn_use_batch`
  - Whether to use batch method for calculation in simple RecurrentLayer.
  - type: bool (default: 0).

* `--prev_batch_state`
  - batch is continue with next batch.
  - type: bool (default: 0).

* `--beam_size`
  - Beam search uses breadth-first search to build its search tree. At each level of the tree, it generates all successors of the states at the current level, sorting them in increasing order of heuristic cost. However, it only stores a predetermined number of best states at each level (called the beam size).
  - type: int32 (default: 1).

* `--diy_beam_search_prob_so`
  - Specify shared dynamic library. It can be defined out of paddle by user.
  - type: string (default: "", null).

## DataProvider

* `--memory_threshold_on_load_data`
  - Stop loading data when memory is not sufficient.
  - type: double (default: 1.0).

## Unit Test

* `--checkgrad_eps`
  - parameter change size for checkgrad.
  - type: double (default: 1e-05).

## Parameter Server and Distributed Communication

* `--start_pserver`
  - Whether to start pserver (parameter server).
  - type: bool (default: 0).

* `--pservers`
  - Comma separated IP addresses of pservers.
  - type: string (default: "127.0.0.1").

* `--port`
  - Listening port for pserver.
  - type: int32 (default: 20134).

* `--ports_num`
  - The ports number for parameter send, increment based on default port number.
  - type: int32 (default: 1).

* `--trainer_id`
  - In distributed training, each trainer must be given an unique id ranging from 0 to num_trainers-1. Trainer 0 is the master trainer. User do not need to care this flag.
  - type: int32 (default: 0).

* `--num_gradient_servers`
  - Numbers of gradient servers. This arguments is set automatically in cluster submitting environment.
  - type: int32 (default: 1).

* `--small_messages`
  - If message size is small, recommend set it True to enable quick ACK and no delay
  - type: bool (default: 0).

* `--sock_send_buf_size`
  - Restrict socket send buffer size. It can reduce network congestion if set carefully.
  - type: int32 (default: 1024 \* 1024 \* 40).

* `--sock_recv_buf_size`
  - Restrict socket recieve buffer size.
  - type: int32 (default: 1024 \* 1024 \* 40).

* `--parameter_block_size`
  - Parameter block size for pserver, will automatically calculate a suitable value if it's not set.
  - type: int32 (default: 0).

* `--parameter_block_size_for_sparse`
  - Parameter block size for sparse update pserver, will automatically calculate a suitable value if it's not set.
  - type: int32 (default: 0).

* `--log_period_server`
  - Log progress every log_period_server batches at pserver end.
  - type: int32 (default: 500).

* `--loadsave_parameters_in_pserver`
  - Load and save parameters in pserver. Only work when parameter set sparse_remote_update.
  - type: bool (default: 0).

* `--pserver_num_threads`
  - number of threads for sync op exec.
  - type: bool (default: 1).

* `--ports_num_for_sparse`
  - The ports number for parameter send, increment based on default (port + ports_num). It is used by sparse Tranning.
  - type: int32 (default: 0).

* `--nics`
  - Network device name for pservers, already set in cluster submitting environment.
  - type: string (default: "xgbe0,xgbe1").

* `--rdma_tcp`
  - Use rdma or tcp transport protocol, already set in cluster submitting environment.
  - type: string (default: "tcp").

## Async SGD
* `--async_count`
  - Defined the asynchronous training length, if 0, then use synchronized training.
  - type: int32 (default: 0).

* `--async_lagged_ratio_min`
  - Control the minimize value of `config_.async_lagged_grad_discard_ratio()`.
  - type: double (default: 1.0).

* `--async_lagged_ratio_default`
  - If async_lagged_grad_discard_ratio is not set in network config, use it as defalut value.
  - type: double (default: 1.5).

## Performance Tuning

* `--log_barrier_abstract`
  - If true, show abstract barrier performance information.
  - type: bool (default: 1).

* `--log_barrier_show_log`
  - If true, always show barrier abstract even with little gap.
  - type: bool (default: 0).

* `--log_barrier_lowest_nodes`
  - How many lowest node will be logged.
  - type: int32 (default: 5).

* `--check_sparse_distribution_in_pserver`
  - Whether to check that the distribution of sparse parameter on all pservers is balanced.
  - type: bool (default: 0).

* `--show_check_sparse_distribution_log`
  - show log details for sparse parameter distribution in pserver.
  - type: bool (default: 0).

* `--check_sparse_distribution_batches`
  - Running sparse parameter distribution check every so many batches.
  - type: int32 (default: 100).

* `--check_sparse_distribution_ratio`
  - If parameters dispatched to different pservers have an unbalanced distribution for check_sparse_distribution_ratio *  check_sparse_distribution_batches times, crash program.
  - type: double (default: 0.6).

* `--check_sparse_distribution_unbalance_degree`
  - The ratio of maximum data size / minimun data size for different pserver.
  - type: double (default: 2).

## Matrix/Vector/RandomNumber
* `--enable_parallel_vector`
  - threshold for enable parallel vector.
  - type: int32 (default: 0).

* `--seed`
  - random number seed. 0 for srand(time)
  - type: int32 (default: 1)

* `--thread_local_rand_use_global_seed`
  - Whether to use global seed in rand of thread local.
  - type: bool (default: 0).
