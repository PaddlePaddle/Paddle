# Performance for Distributed vgg16

## Test Result

### Hardware Infomation

- CPU: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
- cpu MHz		: 2101.000
- cache size	: 20480 KB

### Blas settings

Setting environment variable: `MKL_NUM_THREADS=1`.

### Single Node Single Thread

- Metrics: samples / sec

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
<td> 15.44 </td>
<td> 16.32 </td>
<td> 16.74 </td>
<td> 16.79 </td>
</tr>
<tr>
<td>PaddlePaddle v2  </td>
<td> 15.97 </td>
<td> 17.04 </td>
<td> 17.60 </td>
<td> 17.83 </td>
</tr>
<tr>
<td>TensorFlow </td>
<td> 9.09 </td>
<td> 9.10 </td>
<td> 9.24 </td>
<td> 8.66 </td>
</tr>
</tbody>
</table>


### Different Batch Size

- PServer Count: 10
- Trainer Count: 20
- Metrics: samples / sec

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
<td> 190.20 </td>
<td> 222.15 </td>
<td> 247.40 </td>
<td> 258.18 </td>
</tr>
<tr>
<td>PaddlePaddle v2  </td>
<td> 170.96 </td>
<td> 233.71 </td>
<td> 256.14 </td>
<td> 329.23 </td>
</tr>
<tr>
<td>TensorFlow </td>
<td> - </td>
<td> - </td>
<td> - </td>
<td> - </td>
</tr>
</tbody>
</table>

### Accelerate Rate

- Pserver Count: 20
- Batch Size: 128
- Metrics: samples / sec

<table>
<thead>
<tr>
<th>Trainer Count </th>
<th>20</th>
<th>40</th>
<th>80</th>
<th>100</th>
</tr>
</thead>
<tbody>
<tr>
<td> PaddlePaddle Fluid</td>
<td> 263.29 (78.64%) </td>
<td> 518.80 (77.47%) </td>
<td> 836.26 (62.44%) </td>
<td> 1019.29 (60.89%) </td>
</tr>
<tr>
<td>PaddlePaddle v2 (need more tests)   </td>
<td> 326.85 (92.85%) </td>
<td> 534.58 (75.93%) </td>
<td> 853.30 (60.60%) </td>
<td> 1041.99 (59.20%) </td>
</tr>
<tr>
<td>TensorFlow </td>
<td> - </td>
<td> - </td>
<td> - </td>
<td> - </td>
</tr>
</tbody>
</table>


### Different Pserver Count

- Trainer Count: 60
- Batch Size: 128
- Metrics: samples/ sec

<table>
<thead>
<tr>
<th>PServer Count </th>
<th>3</th>
<th>6</th>
<th>10</th>
<th>20</th>
</tr>
</thead>
<tbody>
<tr>
<td> PaddlePaddle Fluid(should fix in next PR) </td>
<td> 589.1 </td>
<td> 592.6 </td>
<td> 656.4 </td>
<td> 655.8 </td>
</tr>
<tr>
<td>PaddlePaddle v2 (need more tests)   </td>
<td> 593.4 </td>
<td> 791.3 </td>
<td> 729.7 </td>
<td> 821.7 </td>
</tr>
<tr>
<td>TensorFlow </td>
<td> - </td>
<td> - </td>
<td> - </td>
<td> - </td>
</tr>
</tbody>
</table>


*The performance gap between Fuild and v2 comes from the network interference.*


## Steps to Run the Performance Test

1. You must re-compile PaddlePaddle and enable `-DWITH_DISTRIBUTE` to build PaddlePaddle with distributed support.
1. When the build finishes, copy the output `whl` package located under `build/python/dist` to current directory.
1. Run `docker build -t [image:tag] .` to build the docker image and run `docker push [image:tag]` to push the image to reponsitory so kubernetes can find it.
1. Run `kubectl create -f pserver.yaml && kubectl create -f trainer.yaml` to start the job on your kubernetes cluster (you must configure the `kubectl` client before this step).
1. Run `kubectl get po` to get running pods, and run `kubectl logs [podID]` to fetch the pod log of pservers and trainers.

Check the logs for the distributed training progress and analyze the performance.

## Enable Verbos Logs

Edit `pserver.yaml` and `trainer.yaml` and add an environment variable `GLOG_v=3` and `GLOG_logtostderr=1` to see what happend in detail.
