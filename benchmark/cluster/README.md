# Cluster Training Benchmark

## Setup

- Platform
  - Kubernetes: v1.6.2
  - Linux Kernel: v3.10.0

- Resource
  - CPU: 10 Cores per Pod
  - Memory: 5GB per Pod

- Docker Image

  We use different base Docker Image to run the benchmark on Kubernetes:
  - PaddlePaddle v2: paddlepaddle/paddle:0.11.0
  - PaddlePaddle Fluid: paddlepaddle/paddle:[commit-id]
  - TensorFlow: tensorflow/tensorflow:1.5.0-rc0

- Model
  vgg16 is used in this benchmark.

## Cases

- Variable
  - Batch Size of training data.
  - PServer count of the training job.
  - The number of trainers.

- Invariant
  - The resource of trainer/pserver Pod.

### Measure the Performance for Different Batch Size

- PServer Count: 40
- Trainer Count: 100
- Metrics: mini-batch / sec


<table>
<thead>
<tr>
<th>Batch Size </th>
<th> 32</th>
<th>64</th>
<th>128 </th>
<th>256</th>
</tr>
</thead>
<tbody>
<tr>
<td> PaddlePaddle Fluid</td>
<td>-</td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
<tr>
<td>PaddlePaddle v2  </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
<tr>
<td>TensorFlow </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
</tbody>
</table>

### Measure the Performance for Different PServer Count

- Trainer Count: 100
- Batch Size: 64
- Metrics: mini-batch / sec


<table>
<thead>
<tr>
<th>PServer Count  </th>
<th>10</th>
<th>20</th>
<th>40 </th>
<th>60</th>
</tr>
</thead>
<tbody>
<tr>
<td> PaddlePaddle Fluid</td>
<td>-</td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
<tr>
<td>PaddlePaddle v2  </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
<tr>
<td>TensorFlow </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
</tr>
</tbody>
</table>

### Measure Parallel Efficiency By Increasing Trainer Count

- PServer Count: 20
- Batch Size: 64
- Metrics:

$S = \div(T1, TN)$

which S is the ratio of T1 over TN, training time of 1 and N trainers.
The parallel efficiency is:

$E = \div(S, N)$

<table>
<thead>
<tr>
<th>Trainer Counter  </th>
<th>1</th>
<th>10</th>
<th>20 </th>
<th>30</th>
<th>40</th>
<th>50</th>
<th>60 </th>
<th>70</th>
<th>80</th>
<th>90</th>
<th>100 </th>
</tr>
</thead>
<tbody>
<tr>
<td> PaddlePaddle Fluid</td>
<td>-</td>
<td>- </td>
<td>- </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
</tr>
<tr>
<td>PaddlePaddle v2  </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
</tr>
<tr>
<td>TensorFlow </td>
<td>-  </td>
<td>- </td>
<td>-  </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
<td>- </td>
<td>-</td>
<td>- </td>
<td>- </td>
</tr>
</tbody>
</table>


## Reproduce the benchmark

TODO
