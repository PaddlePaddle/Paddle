# Fluid distributed parameter segmentation strategy
In this article, we'll explain the reason and the design of parameters segmentaion when we do pserver-based distributed training with PaddlePaddle Fluid, we will give a case of how this segmentation scheme could be used in python code;

## Background
### Reason for segmentation

In the design of the model, we usually do not limit the size of the parameters used by each layer of the model. Suppose we have 3 parameter servers now and we want to train the following network:

![fluid_3_layer_network](src/fluid_3_layers_network.png)

The fluid.input layer is very wide, causing the w1, b1 parameter dimensions to be very large, reaching 10 * 1000, while the fluid.fc layer is very narrow, resulting in a 1 * 10 dimension of the w2, b2 parameter.

If we simply assign these parameters to the parameter server, the parameter size obtained by each parameter server will not be uniform, and the lightly loaded parameter server will wait for the parameter server with heavy load.
Therefore, for the case of non-uniform size of the parameters, in the Distribute Transpiler, we will segment the parameters of the model and the corresponding gradients into one or more parameter blocks.

## Model Parameter Segmentation Strategy Design
### Segmentation

Take into account the grain size of segmentation, if the segmentation is fine-grained, then the calculation efficiency of the parameter server will be low, but if the segmentation is too coarse-grained, even distribution of the parameters cannot be achieved;
So in order to control the grain size at the time of segmentation, we will calculate two values, the maximum segmentation number and the desired segmentation number for each parameter or gradient:

* The maximum number of cuts

In order to avoid the fine-grained granularity, we have formulated a minimum parameter block size: 8192;
We will round up the result of parameter size / minimum parameter block size, and get the maximum number of segmentation of this parameter;
In the above example, the maximum number of segmentation is 2;

* Expected number of cuts

In order to achieve an even distribution of parameters to each parameter server, we use the total number of parameter servers as the desired number of partitions;
In the above example, the expected number of segmentation is 3;

After calculating the above two values, we will take the smaller of the two values as the final number of cuts, ensuring that the parameters are evenly distributed as far as possible while guaranteeing the minimum granularity;
So in the above example, we will finally divide the parameters into 2 parts;

### Partition

After segment the parameters and gradients into multiple parameter blocks, we also need to evenly partition the parameter blocks to the parameter servers.

Now, we support two simple and effective partition methods: [Round Robin](https://en.wikipedia.org/wiki/Round-robin_scheduling) and [Hash](https://en.wikipedia.org/ Wiki/Hash_function);

In Round Robin mode, we will one-by-one partition the parameter block to the Server;

In Hash mode, we will perform Hash operation on parameter block names and then modulo the total number of parameter servers to obtain a specific parameter server id;

### Overall Segmentation Process

At this point, our strategy for segmenting parameters and gradients is over. For the above example, we will get the segmentation result as shown in the following figure:

![fluid_parameter_slice_up](src/fluid_parameter_slice_up.png)


## Model Parameter Segmentation Use Case
### Distributed Implementation

Specific implementation of PaddlePaddle Fluid distributed training can refer to [Fluid Cluster Train](../../howto/cluster/fluid_cluster_train_cn.md)

### Parameter details
Our main parameter strategy is implemented in [Distribute Transpiler] (https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/transpiler/distribute_transpiler.py), we can use the ```transpile``` method specifies ```slice_var_up=True``` to enable model parameter segmentation, and ```split_method=RoundRobin``` can be used to specify the partition of model parameters. Followings are the sample code:

```python
transpiler.transpile(
	trainer_id=trainer_id,
	slice_var_up=True,
	split_method=RoundRobin,
	pservers=pservers,
	trainers=trainers)
```
