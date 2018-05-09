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


## iperf
- Refrence: 

[iperf3 at 40Gbps and above](https://fasterdata.es.net/performance-testing/network-troubleshooting-tools/iperf/multi-stream-iperf3/)

- Results:
	
<table>
<thead>
<tr>
<th>client </th>
<th>time</th>
<th>speed<br>(Gbits/sec)</th>
</tr>
</thead>
<tbody>
<tr>
<td>s1</td>
<td>37.00-38.00</td>
<td>10.2</td>
</tr>
<tr>
<td>s2</td>
<td>37.00-38.00</td>
<td>14.1</td>
</tr>
<tr>
<td>s3</td>
<td>37.00-38.00</td>
<td>10.4</td>
</tr>
<tr>
<td>s4</td>
<td>37.00-38.00</td>
<td>11.1</td>
</tr>
<tr>
<td>total</td>
<td></td>
<td>45.8</td>
</tr>
</tbody>
</table>

## BRPC

## GRPC 
[Cod is here](https://github.com/gongweibao/tests/tree/develop/grpc_test)

**speed unit: MB/s**
<table>
<thead>
<tr>
<th> </th>
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
<td>pserver(4threads):1<br>client(4threads):1</td>
<td>300</td>
<td>17.49</td>
<td>134.57</td>
<td>195.92</td>
<td>332.66</td>
<td>601.70</td>
<td>438.08</td>
<td>689.00</td>
<td>812.12</td>
<td>1026.10</td>
<td>1101.00</td>
<td>318.85</td>
</tr>
<tr>
<td>server(2threads):4<br>client(1threads):4</td>
<td>512</td>
<td>51</td>
<td>201</td>
<td>400</td>
<td>731</td>
<td>1267</td>
<td>1924</td>
<td>2673</td>
<td>2622</td>
<td>3235</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>server(2threads):4<br>client(4threads):1</td>
<td>512</td>
<td>72.67</td>
<td>166.40</td>
<td>445.62</td>
<td>841.86</td>
<td>1042.73</td>
<td>1612.82</td>
<td>2639.38</td>
<td>2483.70</td>
<td>2714.72</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>server(4threads 4port):1<br>client(4threads):1</td>
<td>300</td>
<td>62.07</td>
<td>198.89</td>
<td>301.83</td>
<td>445.66</td>
<td>1076.86</td>
<td>1494.12</td>
<td>1516.65</td>
<td>2626.78</td>
<td>1820.18</td>
<td>--</td>
<td>--</td>
</tr>
<tr>
<td>server(4threads 2port):1<br>client(4threads):1</td>
<td>512</td>
<td>20.49</td>
<td>94.40</td>
<td>257.80</td>
<td>628.95</td>
<td>684.16</td>
<td>1096.59</td>
<td>1568.67</td>
<td>1497.70</td>
<td>1638.02</td>
<td>--</td>
<td>--</td>
</tr>

</tbody>
</table>

**Notice: GRPC client consume more than 10GB memory in this test when buffer size >= 4MB. And it seems that GRPC creates many threads background.**


## ib\_read\_bw
## MPI 
### With TCP protocal
<table>
<thead>
<tr>
<th>Buffer size </th>
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
</tr>
<tr></tbody>
</table>

### With RDMA
<table>
<thead>
<tr>
<th>Buffer size </th>
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
</tr>
<tr></tbody>
</table>

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

 
