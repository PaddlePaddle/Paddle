# 飞桨安全公告

我们在此定期发布飞桨安全公告。



*注*：我们非常建议飞桨用户阅读和理解[SECURITY_cn.md](../SECURITY_cn.md)所介绍的飞桨安全模型，以便更好地了解此安全公告。


| 安全公告编号                                          | 类型                                                   |    受影响版本    | 报告者                                                             | 备注 |
|-------------------------------------------------|------------------------------------------------------|:-----------:|-----------------------------------------------------------------|----|
| [PDSA-2023-023](./advisory/pdsa-2023-023_cn.md) | Command injection in convert_shape_compare           |   < 2.6.0   | leeya_bug                                                       |    |
| [PDSA-2023-022](./advisory/pdsa-2023-022_cn.md) | FPE in paddle.argmin and paddle.argmax               |   < 2.6.0   | Peng Zhou (zpbrent) from Shanghai University                    |    |
| [PDSA-2023-021](./advisory/pdsa-2023-021_cn.md) | Null pointer dereference in paddle.crop              |   < 2.6.0   | Peng Zhou (zpbrent) from Shanghai University                    |    |
| [PDSA-2023-020](./advisory/pdsa-2023-020_cn.md) | Command injection in _wget_download                  |   < 2.6.0   | huntr.com                                                       |    |
| [PDSA-2023-019](./advisory/pdsa-2023-019_cn.md) | Command injection in get_online_pass_interval        |   < 2.6.0   | huntr.com and leeya_bug                                         |    |
| [PDSA-2023-018](./advisory/pdsa-2023-018_cn.md) | Heap buffer overflow in paddle.repeat_interleave     |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-017](./advisory/pdsa-2023-017_cn.md) | FPE in paddle.amin                                   |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-016](./advisory/pdsa-2023-016_cn.md) | Stack overflow in paddle.linalg.lu_unpack            |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-015](./advisory/pdsa-2023-015_cn.md) | FPE in paddle.lerp                                   |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-014](./advisory/pdsa-2023-014_cn.md) | FPE in paddle.topk                                   |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-013](./advisory/pdsa-2023-013_cn.md) | Stack overflow in paddle.searchsorted                |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-012](./advisory/pdsa-2023-012_cn.md) | Segfault in paddle.put_along_axis                    |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-011](./advisory/pdsa-2023-011_cn.md) | Null pointer dereference in paddle.nextafter         |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-010](./advisory/pdsa-2023-010_cn.md) | Segfault in paddle.mode                              |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-009](./advisory/pdsa-2023-009_cn.md) | FPE in paddle.linalg.eig                             |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-008](./advisory/pdsa-2023-008_cn.md) | Segfault in paddle.dot                               |   < 2.6.0   | Tong Liu of CAS-IIE                                             |    |
| [PDSA-2023-007](./advisory/pdsa-2023-007_cn.md) | FPE in paddle.linalg.matrix_rank                     |   < 2.6.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2023-006](./advisory/pdsa-2023-006_cn.md) | FPE in paddle.nanmedian                              |   < 2.6.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2023-005](./advisory/pdsa-2023-005_cn.md) | Command injection in fs.py                           |   < 2.5.0   | Xiaochen Guo from Huazhong University of Science and Technology |    |
| [PDSA-2023-004](./advisory/pdsa-2023-004_cn.md) | FPE in paddle.linalg.matrix_power                    |   < 2.5.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2023-003](./advisory/pdsa-2023-003_cn.md) | Heap buffer overflow in paddle.trace                 |   < 2.5.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2023-002](./advisory/pdsa-2023-002_cn.md) | Null pointer dereference in paddle.flip              |   < 2.5.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2023-001](./advisory/pdsa-2023-001_cn.md) | Use after free in paddle.diagonal                    |   < 2.5.0   | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2022-002](./advisory/pdsa-2022-002_cn.md) | Code injection in paddle.audio.functional.get_window | = 2.4.0-rc0 | Tong Liu of ShanghaiTech University                             |    |
| [PDSA-2022-001](./advisory/pdsa-2022-001_cn.md) | OOB read in gather_tree                              |    < 2.4    | Wang Xuan(王旋) of Qihoo 360 AIVul Team                           |    |
