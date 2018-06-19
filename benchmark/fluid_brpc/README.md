## Speed
<table>
<thead>
<tr>
<th>framework</th>
<th>model</th>
<th>batch_size</th>
<th>protocal</th>
<th>2 * 2</th>
<th>4 * 4</th>
</tr>
</thead>
<tbody>
<tbody>
<tr>
<td>grpc</td>
<td>Resnet</td>
<td></td>
<td>TCP</td>
<td></td>
<td></td>
</tr>
<tr>
<td>brpc</td>
<td>Resnet</td>
<td></td>
<td>TCP</td>
<td></td>
<td></td>
</tr>
<tr>
<td>grpc</td>
<td>Resnet</td>
<td></td>
<td>RDMA</td>
<td></td>
<td></td>
</tr>
<tr>
<td>brpc</td>
<td>Resnet</td>
<td></td>
<td>RDMA</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

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

## CPU 

```
Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
```

### Docker container
Test is tested in docker container if not fingure out on host.

Run docker with `--network=host`

Docker version

```
Client:
 Version:         1.12.6
 API version:     1.24
 Package version: docker-1.12.6-48.git0fdc778.el7.centos.x86_64
 Go version:      go1.8.3
 Git commit:      0fdc778/1.12.6
 Built:           Thu Sep  7 18:00:07 2017
 OS/Arch:         linux/amd64

Server:
 Version:         1.12.6
 API version:     1.24
 Package version: docker-1.12.6-48.git0fdc778.el7.centos.x86_64
 Go version:      go1.8.3
 Git commit:      0fdc778/1.12.6
 Built:           Thu Sep  7 18:00:07 2017
 OS/Arch:         linux/amd64
```
