<div align="center">

# 🧠 MemoryNav

**視覚記憶ナビゲーションシステム | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-1.9.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v1.9.0)

VPR（視覚的場所認識）とトポロジカルマップに基づくロボット記憶ナビゲーションシステム

[English](README_EN.md) | [中文](README.md) | **日本語** | [한국어](README_KO.md)

</div>

---

## 📖 概要

MemoryNavは移動ロボット向けの視覚記憶ナビゲーションシステムです。4台の全方位魚眼カメラで画像を取得し、VPR技術を用いて事前構築したトポロジカル記憶グラフ上で自己位置推定を行います。Qwen3.5-9B視覚言語モデルをフォールバックとして組み合わせ、「一度行った場所を記憶し、再び歩く」記憶ナビゲーション能力を実現します。

### 主な機能

- **🔍 マルチスキームVPR測位**：4種類のSOTA VPR手法をサポート、設定ファイル一つで切り替え可能
- **🗺️ トポロジカル記憶グラフ**：アノテーションデータからノード-エッジトポロジーを自動構築、最短経路計画（BFS/Dijkstra）をサポート
- **🔄 循環シフトマッチング**：4カメラ循環シフトアルゴリズムによる方向不変の自己位置推定と偏向角推定
- **🎯 DINOv3サブ画像マッチング**：DINOv3高密度パッチ特徴量によるスライディングウィンドウコサイン類似度マッチング、リアルタイムでカメラ画像内のナビゲーション目標を測位
- **💾 マッチングキャッシュ**：信頼度が低い場合は直前の成功結果を自動的に再利用し、ナビゲーションの継続性を向上
- **🔭 Lookahead二重確認**：ステップ切替時にVPR定位と次ステップのサブ画像マッチングを同時検証し、早期advanceを防止
- **📤 統一出力フォーマット**：記憶モードのオン/オフに関わらず一貫したレスポンス形式、常に`pixel_target`を提供
- **🤖 Qwen3.5フォールバックグラウンディング**：VPR/サブ画像マッチング失敗時にQwen3.5-9B VLMへ自動切り替え、中国語のランドマーク名を直接使用
- **📷 魚眼歪み補正**：起動時に`cam/params.yaml`からカメラ内部パラメータを読み込み、VPR・サブ画像マッチング前に入力画像へ円筒投影歪み補正を適用
- **🧭 ピクセル→ロボット座標変換**：正規化された`pixel_target`を円筒角度・カメラ方位角・俯角管線を通じてロボット運動座標`[x_forward, y_lateral, 0.0]`に変換
- **🌐 WebSocketサービス**：リアルタイムストリーミングで画像受信・ナビゲーション指令を返送
- **⚙️ 統一設定管理**：全VPRパラメータを`deploy/vpr_config.yaml`に集中管理

---

## 🏗️ システムアーキテクチャ

```
MemoryNav/
├── cam/                            # 多眼魚眼カメラ ROS2 ノード
│   ├── params.yaml                 # カメラ内部・外部パラメータ (T_ic, intrinsics, distortion)
│   ├── fisheye_undist.h            # GPU加速魚眼歪み補正 (CUDA)
│   ├── main.cc / main.h            # ROS2ノードメインプログラム
│   ├── video.h                     # V4L2ビデオキャプチャ
│   └── tools/                      # スタンドアロンツール（ROS2/CUDA不要）
│       ├── fisheye_undist_cpu.h    # CPU版歪み補正（numpy/cv2移植ベース）
│       ├── fisheye_to_cylindrical.cpp  # 魚眼→円筒CLIツール
│       └── batch_undistort.py      # バッチ歪み補正スクリプト
├── deploy/                         # デプロイモジュール
│   ├── vpr_config.yaml             # VPR統一設定ファイル
│   ├── memory_nav/                 # コア記憶ナビゲーションパッケージ
│   │   ├── vpr_config_loader.py    # 設定ローダー
│   │   ├── memory_models.py        # データモデル (Node, Edge, Plan, VPRResult)
│   │   ├── memory_graph.py         # トポロジーグラフ (BFS/Dijkstra)
│   │   ├── memory_vpr.py           # VPRマッチングエンジン (循環シフト + 順序不変)
│   │   ├── memory_builder.py       # 記憶構築器 (アノテーションデータからトポロジー生成)
│   │   ├── memory_navigator.py     # ナビゲーターメインインターフェース
│   │   ├── sub_image_matcher.py    # サブ画像マッチャー (DINOv3高密度特徴量)
│   │   ├── fisheye_undistort.py    # 🆕 魚眼歪み補正 (cam/tools/fisheye_undist_cpu.hから移植)
│   │   ├── coord_transform.py      # 🆕 ピクセル→ロボット座標変換 (円筒投影完全パイプライン)
│   │   ├── qwen35_point_grounder.py # Qwen3.5グラウンディングラッパー (フォールバックモデル)
│   │   ├── qwen35_grounding_server.py # Qwen3.5サブプロセス推論サーバー
│   │   ├── vpr_factory.py          # VPR抽出器ファクトリー
│   │   ├── anyloc_extractor.py     # AnyLoc (DINOv2 + VLAD)
│   │   ├── megaloc_extractor.py    # MegaLoc (DINOv2 + OT集約)
│   │   ├── effovpr_extractor.py    # EffoVPR (DINOv2 多層CLS)
│   │   └── selavpr_extractor.py    # SelaVPR++ (DINOv2 + MultiConv)
│   ├── ws_proxy_with_memory.py     # WebSocketプロキシサービス (メインエントリー)
│   └── build_memory.sh             # 記憶構築スクリプト
├── internnav/                      # InternNavナビゲーションフレームワーク
├── scripts/
│   └── memory_visualization_server.py  # 記憶グラフ可視化サービス
├── tests/
│   ├── test_memory_nav.py          # 記憶モジュールユニットテスト
│   └── test_memory_ws.py           # WebSocket統合テスト（詳細ログ）
└── docs/
```

---

## 📷 魚眼歪み補正

VPRマッチングおよびサブ画像マッチング前に、4チャンネルの魚眼画像に対して自動的に円筒投影歪み補正を実施します。

### 動作原理

1. 起動時に`cam/params.yaml`から各カメラの内部パラメータ（`xi, fx, fy, cx, cy`）と歪み係数（`k1, k2, p1, p2`）を読み込む
2. カメラごとに1回だけ円筒投影remapテーブルを事前計算（`pitch_up`オフセット込み）
3. フレームごとの推論前に`cv2.remap`を適用、計算コストは極めて低い
4. `cam/params.yaml`が存在しない場合は歪み補正をスキップ、サービスは正常起動

```python
from deploy.memory_nav.fisheye_undistort import FisheyeUndistorter

undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
perspective_images = undistorter.undistort_batch(camera_images)
```

---

## 🧭 ピクセル→ロボット座標変換

正規化された`pixel_target: [x_norm, y_norm]`を物理パイプラインを通じてロボット運動座標`[x_forward, y_lateral, 0.0]`に変換します。

### 変換パイプライン

```
x_norm → 円筒水平角 → + カメラ方位角 → グローバルyaw
y_norm → 円筒垂直角 → 俯角 → 距離推定（カメラ高さ + pitch_up）
yaw + distance → (x_forward, y_lateral)
```

### カメラ方位角（`cam/params.yaml` T_icから算出）

| カメラ | 方位角 |
|--------|--------|
| camera_1 | +39.42° |
| camera_2 | −35.84° |
| camera_3 | −142.04° |
| camera_4 | +143.52° |

---

## 🎯 サブ画像マッチングナビゲーション

**DINOv3**高密度パッチ特徴量によるサブ画像マッチング：

### 動作原理

1. **記憶構築時**：各エッジに`camera_name`（目標カメラ）と`crop_image`（注意クロップ）をアノテーション
2. **ナビゲーション実行時**：記憶からcropサブ画像を取得し、リアルタイムカメラ画像に対して高密度特徴量マッチングを実施
3. **目標測位**：DINOv3 ViT-B/16で高密度パッチトークンを抽出 → スライディングウィンドウ + unfold加速 → コサイン類似度最大位置 → `pixel_target`として出力
4. **マッチング閾値**：信頼度 ≥ `SUB_MATCH_CONFIDENCE_THRESHOLD`（現在0.65）で成功
5. **キャッシュ**：低信頼度フレームでは直前の成功結果を再利用、ステップ切り替え時にクリア
6. **フォールバック**：キャッシュなしの場合、記憶内の`pixel_box`を推定値として使用

---

## ✨ VPR手法比較

| 手法 | キー | 発表 | 特徴次元 | バックボーン | 特徴 |
|------|------|------|---------|----------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **推奨**、ハッシング+リランク |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | 総合最強、複数データセットSOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 多層CLS | 軽量・高速、リアルタイム向き |
| **AnyLoc** | `anyloc` | RA-L 2023 | 可変 | DINOv2-B + VLAD | クラシック、安定 |

---

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### カメラセットアップ（任意）

実機カメラがある場合、キャリブレーションファイルを`cam/params.yaml`に配置してください。起動時に自動的に魚眼歪み補正が有効になります。

### 記憶構築

```bash
bash deploy/build_memory.sh
```

### ナビゲーションサービス起動

```bash
python deploy/ws_proxy_with_memory.py
```

---

## 📡 WebSocketプロトコル

### リクエスト

```json
{
    "id": "robot_01",
    "pts": 1709558400,
    "task": "受付に案内してください",
    "images": {
        "camera_1": "<base64>",
        "camera_2": "<base64>",
        "camera_3": "<base64>",
        "camera_4": "<base64>"
    }
}
```

### レスポンス

```json
{
    "status": "success",
    "id": "robot_01",
    "task_status": "executing",
    "action": [[0.5, -0.1, 0.0]],
    "pixel_target": [0.485, 0.521],
    "memory_active": true,
    "camera_name": "camera_2",
    "landmark_name": "エレベーター",
    "memory_info": {
        "coord_transform": {
            "yaw_global_deg": -12.3,
            "depression_deg": 8.5,
            "distance": 2.4,
            "elapsed_ms": 0.3
        }
    }
}
```

---

## 📋 更新履歴

### v1.9.0

- **🔭 Lookahead二重確認**：ステップ切替条件をVPR + 次ステップサブ画像マッチングの二重確認に強化
  - 毎フレーム現在のステップと次ステップを同時にサブ画像マッチング
  - VPRが目標ノードに一致しても、次ステップの一致が成功しなければadvanceしない
  - 最終ステップはlookahead不要で直接advance
- **🎯 サブ画像マッチング閾値統一**：`SUB_MATCH_CONFIDENCE_THRESHOLD`を唯一の真実のソースとして全パイプラインに伝播
- **📊 テストログ強化**：`la_conf`列追加、Lookahead統計セクション追加

### v1.8.0

- **🆕 魚眼歪み補正**：`deploy/memory_nav/fisheye_undistort.py`を追加（`cam/tools/fisheye_undist_cpu.h`から移植）
- **🆕 ピクセル→ロボット座標変換**：`deploy/memory_nav/coord_transform.py`を追加（完全円筒投影パイプライン）
- **🆕 cam/ディレクトリ**：多眼魚眼カメラROS2ノードのソースコードと`params.yaml`を追加

### v1.7.0

- **Qwen3.5フォールバックグラウンディング**：InternVLAの代わりにQwen3.5-9B VLMを採用

### v1.6.0

- **サブ画像マッチング簡略化**：DINOv3高密度特徴量マッチングのみを維持

### v1.5.0

- **統一出力フォーマット**：常に`pixel_target`を含む一貫したレスポンス形式

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) を採用しています。
