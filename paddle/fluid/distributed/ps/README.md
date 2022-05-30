# 目录说明

Table: for param storage and update
-----MemorySparseTable: table for sparse param, used in cpu async mode
-----MemoryDenseTable: table for dense param, used in cpu async/geo mode
-----MemorySparseGeoTable: table for sparse param, used in cpu async mode
-----CommonGraphTable: table used for graph learning
-----BarrierTable: table for barrier function, used in cpu sync mode
-----TensorTable: table which run program, used for learning rate decay only

ValueAccessor: for pull param and push gradient
-----CtrCommonAccessor: pull/push value with show/click, float type
-----CtrDoubleAccessor: same as CtrCommonAccessor, other than show/click with double type
-----SparseAccessor: used for common embedding, pull value without show/click, push value with show/click
-----CommMergeAccessor: used for dense table only, for get param dim

PsService(proto): for server to handle request
-----PsBaseService
----------BrpcPsService: for cpu dnn training task
----------GraphBrpcService: for graph learning
-----HeterService: for dnn training task with heterogeneous computing resources

PSServer: recv request from trainer and handle it by service
-----BrpcPsServer: for cpu dnn training task
-----GraphBrpcServer: for graph learning
-----PsLocalServer: for GpuPS

HeterServer: for HeterPS

PSClient: pull param and push gradient for trainer
-----BrpcPsClient: for cpu dnn training task
----------GraphBrpcClient: for graph learning
-----PsLocalClient: for GpuPS

HeterClient: for HeterPS

PSCore: Wrapper for InitServer

GraphPyService: for graph learning
