# Throughput of a program on the 100Gb network.
## Motivation
Throughput is the main feature of our choice of a framework.
 
   - Compared with the delay,we are more concerned with the throughput of the framework.   
   - Benchmark of framework maybe not true in our envirionment as they said.
   - It's not useful if we can't make use of a framework's potential: we maybe not expert of the framework.

### Some common questions:
- Can data transformation use TCP protocol use full of a 100Gb network?
    - Is RDMA necessary?
    - Is MPI(MPI Send and MPI Receive) necessary?
- What's the TCP program's throughput upper limit?
    - How to tune the TCP program's throughput?
- What's the benchmark of GPU direct? Is it high cost-effective?

## Hardware Infomation

## BRPC

## GRPC 
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
<td>1 * 1</td>
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
<tr>
<td>2 * 2</td>
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
<tr>
<td>4 * 4</td>
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
</tbody>
</table>

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
</tbody>
</table>

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
<td>1 * 1</td>
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
<tr>
<td>2 * 2</td>
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
<tr>
<td>4 * 4</td>
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
</tbody>
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
<td>1 * 1</td>
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
<tr>
<td>2 * 2</td>
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
<tr>
<td>4 * 4</td>
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
</tbody>
</table>

### With RADMA + GPU direct
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
<td>1 * 1</td>
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
<tr>
<td>2 * 2</td>
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
<tr>
<td>4 * 4</td>
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
</tbody>
</table>


## Conclution

 
