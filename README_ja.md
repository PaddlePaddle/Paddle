<p align="center">
<img align="center" src="./doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

[English](./README.md) | [简体中文](./README_cn.md) | 日本語

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](./LICENSE)
![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FPaddlePaddle)

PaddlePaddle GitHub へようこそ。

PaddlePaddle は中国初の独立系 R&D ディープラーニングプラットフォームとして、2016年からプロのコミュニティに正式にオープンソース化されました。コアとなる深層学習フレームワーク、基本モデルライブラリ、エンドツーエンドの開発キット、ツール＆コンポーネント、さらにサービスプラットフォームを網羅する、高度な技術と豊富な機能を備えた産業プラットフォームです。
PaddlePaddle は産業での実践から生まれ、産業化への専念とコミットメントを重ねてきました。製造業、農業、エンタープライズサービスなど幅広い分野で採用され、2,333 万人以上の開発者、76 万社以上の企業に利用され、110 万以上のモデルを生み出しています。こうした強みにより、PaddlePaddle は AI の商用化に取り組むパートナーをますます多く支援しています。

## インストール

### PaddlePaddle の最新リリース: 3.3

私たちのビジョンは、PaddlePaddle を通じて、誰もが深層学習を行えるようにすることです。
PaddlePaddle の最新機能を追跡するために、私たちの[リリースのお知らせ](https://github.com/PaddlePaddle/Paddle/releases)を参照してください。

### 最新の安定版またはナイトリービルドをインストールする

インストール方法については、[クイックインストール](https://www.paddlepaddle.org.cn/install/quick)をご覧ください

## **PaddlePaddle 新世代フレームワーク 3.2**

- **動的グラフと静的グラフの統合および自動並列**

  単一カード構成を基に最小限のテンソル分割アノテーションを追加するだけで、PaddlePaddle は最も効率的な分散並列戦略を自動的に探索します。これにより、産業開発とトレーニングのコストを大幅に削減し、開発者がモデルとアルゴリズムの革新により集中できるようにします。

- **大規模モデルのトレーニングと推論の統合**

  同じフレームワークでトレーニングと推論の両方をサポートし、コード再利用と両段階のシームレスな連携を実現します。大規模モデルのワークフロー全体に統一された開発体験と最大限のトレーニング効率を提供し、産業界に優れた開発体験をもたらします。

- **科学計算向け高階微分**

  高階自動微分、複素数演算、フーリエ変換、コンパイル最適化、分散トレーニング支援などの機能を提供します。数学、力学、材料科学、気象学、生物学などの分野での科学的探索を促進し、微分方程式の求解速度を大きく向上させます。

- **ニューラルネットワークコンパイラ**

  統合フレームワーク設計を採用し、生成モデルや科学計算モデルを含む多様なモデルに対して効率的なトレーニングと柔軟な推論をサポートします。計算の柔軟性と高性能の間で有効なバランスを実現し、性能最適化のコストを大幅に低減します。

- **異種マルチチップ適応**

  複数のハードウェアタイプに対応する成熟した完全な統一適応ソリューションを備えています。標準化されたインターフェースを通じて、チップソフトウェアスタックごとに異なる開発インターフェースの差異を抽象化し、プラグイン可能なアーキテクチャを実現します。

## ドキュメント

[英語](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)と
[中国語](https://www.paddlepaddle.org.cn/documentation/docs/zh/guide/index_cn.html)のドキュメントを提供しています。

- [ガイド](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

  PaddlePaddle でディープラーニングの基本を実装する方法から始めてみてはいかがでしょうか。

- [プラクティス](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  Paddle を使ってモデルを構築し、ディープラーニングタスクをより効率的に実行しましょう。

- [API リファレンス](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   新しい API により、より短時間のプログラムが可能となりました。

- [コントリビュート方法](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/08_contribution/index_en.html)

   皆様のご投稿に感謝いたします！

## オープンソースコミュニティ

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): バグレポート、機能リクエスト、インストールに関する問題、使用方法に関する問題など。
- 経験豊富なコミュニティメンバーによるさまざまなレベルのメンタリングを提供するコントリビューションイベントが多数あります。固定された issues のイベントを確認し、ぜひ参加をご検討ください。
- コミュニティブログ: <https://pfcc.blog/>
- PaddlePaddle コミュニティの詳細は [community](https://github.com/PaddlePaddle/community) をご覧ください。

## 著作権とライセンス

PaddlePaddle は [Apache-2.0 license](./LICENSE) の下で提供されています。
