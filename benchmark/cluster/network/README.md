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

 
