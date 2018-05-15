# Throughput of a program on the 100Gb network.
## Motivation
Throughput is the main feature of our choice of a framework. And We need framework of multi-thread model to use full of network throughput.However,multi-process sometimes can be used to test the limit of TCP on network hardware.
 
   - Compared with the delay,we are more concerned with the throughput of the framework.   
   - Benchmark of framework maybe not true in our envirionment as they said.
   - It's not useful if we can't make use of a framework's potential: we maybe not expert of the framework.
   -  

### Some common questions:
- Can data transformation use TCP protocol use full of a 100Gb network?
    - Is RDMA necessary?
    - Is MPI(MPI Send and MPI Receive) necessary?
- What's the TCP program's throughput upper limit?
    - How to tune the TCP program's throughput?
- What's the benchmark of GPU direct? Is it high cost-effective?

## Hardware Infomation
### Network card
```
driver: mlx5_core
version: 4.3-1.0.1
firmware-version: 12.17.1010 (MT_2140110033)
bus-info: 0000:82:00.0
supports-statistics: yes
supports-test: yes
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: yes
```

NOTE: this config will cause performancy decay:

```
driver: mlx5_core
version: 3.0-1 (January 2015)
firmware-version: 12.17.1010
bus-info: 0000:82:00.0
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
```

### CPU 

```
Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
```


## iperf
- Refrence: 

[iperf3 at 40Gbps and above](https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/multi-stream-iperf3/)

- iperf2: 84Gb - 94Gb
 
```
[  4]  0.0-10.0 sec  9.64 GBytes  8.28 Gbits/sec
[  3]  0.0-10.0 sec  10.1 GBytes  8.65 Gbits/sec
[  5]  0.0-10.0 sec  9.95 GBytes  8.55 Gbits/sec
[  6]  0.0-10.0 sec  9.50 GBytes  8.16 Gbits/sec
[  9]  0.0-10.0 sec  9.56 GBytes  8.21 Gbits/sec
[  8]  0.0-10.0 sec  11.1 GBytes  9.51 Gbits/sec
[  7]  0.0-10.0 sec  12.3 GBytes  10.6 Gbits/sec
[ 11]  0.0-10.0 sec  13.7 GBytes  11.7 Gbits/sec
[ 12]  0.0-10.0 sec  13.4 GBytes  11.5 Gbits/sec
[ 10]  0.0-10.0 sec  10.3 GBytes  8.86 Gbits/sec
[SUM]  0.0-10.0 sec   110 GBytes  94.0 Gbits/sec
```

## evpp

[Code](https://github.com/Qihoo360/evpp/tree/master/benchmark/throughput)

- server: benchmark_pingpong_server 9099 12
- client: benchmark_pingpong_client 192.168.16.30 9099 12 1048976 12 10

```
W0515 03:08:56.568994    98 client.cc:127] name=./benchmark_pingpong_client 88730614724 total bytes read
W0515 03:08:56.569007    98 client.cc:128] name=./benchmark_pingpong_client 1215539 total messages read
W0515 03:08:56.569015    98 client.cc:129] name=./benchmark_pingpong_client 72996.9 average message size
W0515 03:08:56.569452    98 client.cc:130] name=./benchmark_pingpong_client 8462.01 MiB/s throughput
```

## muduo
- server: ./pingpong_server 0.0.0.1 9912 16
- client: ./pingpong_client 192.168.16.30 9912 16 1048976 16 10

```
20180515 03:07:13.052579Z    66 INFO  pid = 66, tid = 66 - client.cc:199
20180515 03:07:13.081942Z    67 WARN  all connected - client.cc:125
20180515 03:07:23.052766Z    66 WARN  stop - client.cc:162
20180515 03:07:23.062692Z    82 WARN  all disconnected - client.cc:133
20180515 03:07:23.062721Z    82 WARN  90249274920 total bytes read - client.cc:143
20180515 03:07:23.062728Z    82 WARN  1293237 total messages read - client.cc:144
20180515 03:07:23.062734Z    82 WARN  69785.5651516 average message size - client.cc:145
20180515 03:07:23.062751Z    82 WARN  8606.84155655 MiB/s throughput - client.cc:147
```

## BRPC

## GRPC 
[Cod is here](https://github.com/gongweibao/tests/tree/develop/grpc_test)

**speed unit: MB/s**
<table>
<thead>
<tr>
<th>server</th>
<th>port</th>
<th>client</th>
<th>loop times</th>
<th>4K</th>
<th>16K</th>
<th>32K</th>
<th>64K</th>
<th>128K</th>
<th>256K</th>
<th>512K</th>
<th>1M</th>
<th>2M</th>
<th>4M</th>
<th>8M</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>1</td>
<td>1</td>
<td>300</td>
<td>42.77</td>
<td>242.27</td>
<td>375.31</td>
<td>422.04</td>
<td>460.51</td>
<td>723.10</td>
<td>1121.15</td>
<td>724.99</td>
<td>1272.53</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>1</td>
<td>4</td>
<td>1</td>
<td>300</td>
<td>64.96</td>
<td>327.71</td>
<td>689.90</td>
<td>1374.64</td>
<td>2162.38</td>
<td>1585.05</td>
<td>3299.54</td>
<td>2413.42</td>
<td>3053.73</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>1</td>
<td>2</td>
<td>1</td>
<td>300</td>
<td>66.30</td>
<td>328.59</td>
<td>388.92</td>
<td>486.26</td>
<td>1510.74</td>
<td>1527.20</td>
<td>1807.75</td>
<td>1986.17</td>
<td>1793.20</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>4</td>
<td>1</td>
<td>1</td>
<td>300</td>
<td>57.35</td>
<td>419.05</td>
<td>829.98</td>
<td>1640.64</td>
<td>1842.19</td>
<td>2322.42</td>
<td>2505.90</td>
<td>2747.26</td>
<td>2826.74</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>4</td>
<td>1</td>
<td>4</td>
<td>300</td>
<td>167</td>
<td>625</td>
<td>1136</td>
<td>1923</td>
<td>2586</td>
<td>3296</td>
<td>3614</td>
<td>3973</td>
<td>3864</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>4</td>
<td>1</td>
<td>1</td>
<td>300</td>
<td>57.35</td>
<td>419.05</td>
<td>829.98</td>
<td>1640.64</td>
<td>1842.19</td>
<td>2322.42</td>
<td>2505.90</td>
<td>2747.26</td>
<td>2826.74</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>10</td>
<td>1</td>
<td>10</td>
<td>300</td>
<td>41</td>
<td>4</td>
<td>637</td>
<td>172</td>
<td>1000</td>
<td>3024</td>
<td>3937</td>
<td>3768</td>
<td>4095</td>
<td>--</td>
<td>--</td>
</tr>
</tbody>
</table>

**Notice: GRPC client consume more than 10GB memory in this test when buffer size >= 4MB. And it seems that GRPC creates many threads background.**


## ib\_read\_bw
## MPI 

code: [MVAPICH: MPI over InfiniBand, Omni-Path, Ethernet/iWARP, and RoCE](http://mvapich.cse.ohio-state.edu/benchmarks/)

### With TCP protocal: by hanqing

Single thread + single link?

```
# OSU MPI-CUDA Bandwidth Test v5.4.1
# Send Buffer on HOST (H) and Receive Buffer on HOST (H)
# Size      Bandwidth (MB/s)
1                       0.41
2                       0.81
4                       1.60
8                       3.03
16                      6.63
32                     12.49
64                     22.89
128                    48.93
256                    78.92
512                   177.38
1024                  325.69
2048                  588.07
4096                  979.27
8192                 1578.38
16384                2565.40
32768                2491.85
65536                3344.32
131072               3877.29
262144               4344.95
524288               4542.15
1048576              4420.13
2097152              4369.98
4194304              4199.28
```

### With RDMA: by hanqing


```
# Send Buffer on DEVICE (D) and Receive Buffer on DEVICE (D)
# Size      Bandwidth (MB/s)
1                       0.08
2                       0.16
4                       0.31
8                       0.62
16                      1.23
32                      2.51
64                      4.96
128                     9.89
256                    19.91
512                    40.94
1024                   81.97
2048                  162.98
4096                  302.72
8192                  555.64
16384                1454.73
32768                2949.01
65536                5035.35
131072               4910.93
262144               6172.67
524288               7676.30
1048576              8453.93
2097152              8897.36
4194304              9170.23
```

### With RADMA + GPU direct
<table>
<thead>
<tr>
<th></th>
<th>4K</th>
<th>16K</th>
<th>32K</th>
<th>64K</th>
<th>128K</th>
<th>256K</th>
<th>512K</th>
<th>1M</th>
<th>2M</th>
<th>4M</th>
<th>8M</th>
</tr>
</thead>
<tbody>
<tr>
<td>server(4threads)<br>client(4threads)</td>
<td>  </td>
<td>  </td>
<td>  </td>
<td>  </td>
<td>  </td>
<td>  </td>
<td>  </td>
<td> </td>
<td> </td>
<td> </td>
<td> </td>
</tr></tbody>
</table>


## Conclution

 
