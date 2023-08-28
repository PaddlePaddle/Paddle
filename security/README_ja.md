# PaddlePaddle セキュリティ勧告

PaddlePaddle の使用に関するセキュリティ勧告を定期的に発表しています。



*注*: これらのセキュリティ勧告と併せ、PaddlePaddle ユーザーには [SECURITY.md](../SECURITY_ja.md) に記載されている PaddlePaddle のセキュリティモデルを読み、理解することを強くお勧めします。


| アドバイザリー番号                              | タイプ                                                 | 対象バージョン | 報告者                                                      | 追加情報 |
|----------------------------------------------|------------------------------------------------------|:-----------------:|------------------------------------------------------------------|------------------------|
| [PDSA-2023-005](./advisory/pdsa-2023-005.md) | Command injection in fs.py                           |      < 2.5.0      | Xiaochen Guo from Huazhong University of Science and Technology  |                        |
| [PDSA-2023-004](./advisory/pdsa-2023-004.md) | FPE in paddle.linalg.matrix_power                    |      < 2.5.0      | Tong Liu of ShanghaiTech University                              |                        |
| [PDSA-2023-003](./advisory/pdsa-2023-003.md) | Heap buffer overflow in paddle.trace                 |      < 2.5.0      | Tong Liu of ShanghaiTech University                              |                        |
| [PDSA-2023-002](./advisory/pdsa-2023-002.md) | Null pointer dereference in paddle.flip              |      < 2.5.0      | Tong Liu of ShanghaiTech University                              |                        |
| [PDSA-2023-001](./advisory/pdsa-2023-001.md) | Use after free in paddle.diagonal                    |      < 2.5.0      | Tong Liu of ShanghaiTech University                              |                        |
| [PDSA-2022-002](./advisory/pdsa-2022-002.md) | Code injection in paddle.audio.functional.get_window |    = 2.4.0-rc0    | Tong Liu of ShanghaiTech University                              |                        |
| [PDSA-2022-001](./advisory/pdsa-2022-001.md) | OOB read in gather_tree                              |       < 2.4       | Wang Xuan(王旋) of Qihoo 360 AIVul Team                            |                        |
