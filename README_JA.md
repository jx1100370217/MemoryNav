<div align="center">

# 🧠 MemoryNav

**視覚記憶ナビゲーションシステム | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.2.0)

VPR（視覚的場所認識）とトポロジカルマップに基づくロボット記憶ナビゲーションシステム

[English](README_EN.md) | [中文](README.md) | **日本語** | [한국어](README_KO.md)

</div>

---

## 📖 概要

MemoryNavは移動ロボット向けの視覚記憶ナビゲーションシステムです。4台の全方位魚眼カメラで画像を取得し、VPR技術を用いて事前構築したトポロジカル記憶グラフ上で自己位置推定を行います。YOLOv8nによる視覚遮蔽検出とQwen3.5-9B視覚言語モデルのフォールバックを組み合わせ、「一度行った場所を記憶し、再び歩く」記憶ナビゲーション能力を実現します。

### 主な機能

- **🗺️ 自動建図**：3段階Pipeline（VPRノード作成→語義増補→接続生成）で画像シーケンスからトポロジカルナビゲーショングラフを全自動生成、手動アノテーション不要
- **🔍 マルチスキームVPR測位**：4種類のSOTA VPR手法をサポート、設定ファイル一つで切り替え可能
- **🗺️ トポロジカル記憶グラフ**：アノテーションデータからノード-エッジトポロジーを自動構築、最短経路計画（BFS/Dijkstra）をサポート
- **🔄 循環シフトマッチング**：4カメラ循環シフトアルゴリズムによる方向不変の自己位置推定と偏向角推定
- **🎯 DINOv3サブ画像マッチング**：DINOv3高密度パッチ特徴量による3段階カスケードマッチング（small→mid→big）＋全カメラ走査
- **💾 フレーム間キャッシュ再利用**：DINOv2 VPR特徴量によるフレーム間類似度（追加推論コストゼロ）、マッチング失敗時に前フレームの成功結果をインテリジェントに再利用
- **🔭 Lookahead二重確認**：ステップ切替時にVPR定位と次ステップのサブ画像マッチングを同時検証し、早期advanceを防止
- **📤 統一出力フォーマット**：記憶モードのオン/オフに関わらず一貫したレスポンス形式、常に`pixel_target`を提供
- **🚧 YOLOv8n遮蔽検出**：サブ画像マッチング失敗時にカメラ画面の遮蔽を自動検出（VPR結果とは独立）、遮蔽時はその場で待機し解消後にナビゲーション再開
- **🤖 Qwen3.5フォールバック**：2段階推論（存在確認＋条件付きグラウンディング）によるQwen3.5-9B VLMフォールバック
- **📷 魚眼歪み補正**：起動時に`cam/params.yaml`からカメラ内部パラメータを読み込み、円筒投影歪み補正を適用
- **🧭 ピクセル→ロボット座標変換**：正規化された`pixel_target`を完全物理パイプラインでロボット運動座標に変換
- **🔄 側面カメラ回転処理**：camera_3/camera_4マッチング成功時に、目標方向への回転動作を自動出力
- **🌐 WebSocketサービス**：リアルタイムストリーミングで画像受信・ナビゲーション指令を返送
- **⚙️ 統一設定管理**：全VPRパラメータを`deploy/vpr_config.yaml`に集中管理

---

## 🏗️ システムアーキテクチャ

```
MemoryNav/
├── memory_nav/                     # コア記憶ナビゲーションモジュール
│   ├── memory_navigator.py         # ナビゲーターメインインターフェース
│   ├── memory_models.py            # データモデル (Node, Edge, Plan, VPRResult)
│   ├── memory_graph.py             # トポロジーグラフ (BFS/Dijkstra)
│   ├── memory_vpr.py               # VPRマッチングエンジン (循環シフト + 順序不変)
│   ├── memory_builder.py           # 記憶構築器
│   ├── sub_image_matcher.py        # サブ画像マッチャー (DINOv3高密度特徴量)
│   ├── occlusion_detector.py       # YOLOv8n遮蔽検出器
│   ├── fisheye_undistort.py        # 魚眼歪み補正 (円筒投影)
│   ├── coord_transform.py          # ピクセル→ロボット座標変換
│   ├── qwen35_point_grounder.py    # Qwen3.5グラウンディング (フォールバック)
│   ├── qwen35_grounding_server.py  # Qwen3.5サブプロセス推論サーバー
│   ├── vpr_factory.py              # VPR抽出器ファクトリー
│   ├── vpr_config_loader.py        # 設定ローダー
│   └── selavpr_model/              # SelaVPR++モデルコード
├── deploy/                         # デプロイエントリー
│   ├── ws_proxy_with_memory.py     # WebSocketプロキシサービス (メインエントリー)
│   ├── vpr_config.yaml             # VPR統一設定ファイル
│   ├── build_memory.sh             # 記憶構築スクリプト
│   └── start_server.sh             # サーバー起動スクリプト
├── cam/                            # 多眼魚眼カメラ
│   ├── params.yaml                 # カメラパラメータ
│   └── tools/                      # スタンドアロンツール
├── scripts/
│   └── memory_visualization_server.py  # 可視化サービス (サブ画像 + 打点 + 遮蔽検出)
├── pretrained/                     # 事前学習モデル (YOLOv8n, DINOv3等)
├── merged_labeled_data/            # 記憶アノテーションデータ
├── auto_mapper/                    # 自動建図モジュール
│   ├── run_auto_map.py             # エントリースクリプト
│   ├── auto_mapper_core.py         # コアコントローラー (3段階Pipeline)
│   ├── node_distance_estimator.py  # VPRノード距離推定
│   ├── auto_landmark_namer.py      # Qwen3.5シーン命名 (vLLM)
│   ├── semantic_node_detector.py   # ドアプレート/標識文字認識
│   ├── auto_node_generator.py      # ノードディレクトリ・メタデータ生成
│   ├── auto_sub_image_extractor.py # グラウンディングcrop + 廊下フレームマッチング
│   └── validate_output.py         # 出力フォーマット検証
├── tests/
│   └── test_memory_ws.py           # WebSocket統合テスト
└── docs/
```

---

## 📷 魚眼歪み補正

VPRマッチングおよびサブ画像マッチング前に、4チャンネルの魚眼画像に対して自動的に円筒投影歪み補正を実施します。

1. 起動時に`cam/params.yaml`から各カメラの内部パラメータ（`xi, fx, fy, cx, cy`）と歪み係数（`k1, k2, p1, p2`）を読み込む
2. カメラごとに1回だけ円筒投影remapテーブルを事前計算（`pitch_up`オフセット込み）
3. フレームごとの推論前に`cv2.remap`を適用、計算コストは極めて低い
4. `cam/params.yaml`が存在しない場合は歪み補正をスキップ、サービスは正常起動

---

## 🧭 ピクセル→ロボット座標変換

正規化された`pixel_target: [x_norm, y_norm]`を物理パイプラインを通じてロボット運動座標`[x_forward, y_lateral, 0.0]`に変換します。

### 側面カメラ回転処理

camera_3またはcamera_4（後方向き）がマッチングに成功した場合、前進動作ではなく座標変換で実際のyaw角を計算し、その場での回転動作`[0, 0, yaw_rad]`を出力してロボットを目標方向に向けます。

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

1. **記憶構築時**：各エッジに`camera_name`と3段階`crop_image`（big/mid/small）をアノテーション
2. **ナビゲーション実行時**：全4カメラをsmall→mid→big 3段階カスケードで走査し、グローバル最高confの結果を選択
3. **目標測位**：DINOv3 ViT-B/16で高密度パッチトークンを抽出 → スライディングウィンドウ + unfold加速 → コサイン類似度最大位置 → `pixel_target`として出力
4. **マッチング閾値**：信頼度 ≥ **0.60**（`SUB_MATCH_CONFIDENCE_THRESHOLD`）で成功
5. **フレーム間キャッシュ**：低信頼度フレームではDINOv2 VPR特徴量による類似度（閾値**0.70**）で前フレーム結果の再利用を判定、ステップ切り替え時にクリア
6. **フォールバック**：キャッシュなしの場合、Qwen3.5兜底グラウンディングをトリガー

---

## 🚧 遮蔽検出

サブ画像マッチング失敗時（VPR結果とは独立）に、注意カメラの遮蔽検出を自動実行：

1. **トリガー**：サブ画像マッチング失敗で即トリガー — VPR結果に依存しない
2. **カメラ選択**：サブ画像マッチングスコア最高（閾値未満）のcameraを使用
3. **YOLOv8n推論**：近距離前景物体（person、backpack、umbrella、handbag、suitcase）を検出、bbox面積比を計算
4. **遮蔽判定**：単一遮蔽物面積比 ≥ **25%**（デフォルト）→ 遮蔽と判定
5. **遮蔽時**：`action: [0, 0, 0]`（その場で待機）、サブ画像キャッシュクリア
6. **遮蔽なし**：Qwen3.5打点（固定「通路中央+奥行き」経路探索）でフォールバックナビゲーション

### ナビゲーション決定フロー

```
フレームごと:
  ├─ サブ画像マッチング (全4カメラ × 3段階cascade)
  ├─ Lookahead 次ステップサブ画像マッチング
  │
  ├─ サブ画像マッチング失敗時:
  │   ├─ YOLOv8n遮蔽検出 (最高スコアcameraで)
  │   │   ├─ 遮蔽 → action=[0,0,0] 待機、キャッシュクリア
  │   │   └─ 遮蔽なし → Qwen3.5打点 → 失敗なら記憶再送
  │
  ├─ VPR成功:
  │   ├─ 目標ノード + sim≥0.70:
  │   │   ├─ 最終ステップ → 直接advance
  │   │   ├─ 次ステップマッチOK → Lookahead確認 → advance
  │   │   └─ 次ステップマッチNG → VPR HELD
  │   └─ 他ノード / sim<0.70 → 現ステップ継続
  │
  └─ VPR失敗:
      ├─ サブマッチOK → サブマッチ結果でナビ継続
      ├─ サブマッチNG + Qwen3.5 OK → 打点結果でナビ
      └─ 全失敗 → 記憶引導再送
```

---

## ✨ VPR手法比較

| 手法 | キー | 発表 | 特徴次元 | バックボーン | 特徴 |
|------|------|------|---------|----------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **推奨**、ハッシング+リランク |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | 総合最強、複数データセットSOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 多層CLS | 軽量・高速 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 可変 | DINOv2-B + VLAD | クラシック、安定 |

---

## 🗺️ 自動建図

自動建図モジュール（`auto_mapper/`）は、ロボットが撮影した画像シーケンスからトポロジカルナビゲーショングラフを全自動生成します。手動アノテーション不要です。

### 3段階Pipeline

```
Phase 1: VPR 粗粒度ノード作成
  ├─ フレーム順に4カメラ画像をスキャン
  ├─ VPR特徴量抽出 → 最近ノードとの類似度比較
  ├─ 類似度 < 閾値(0.70) → 新ノード作成
  └─ Qwen3.5 VLMで自動命名(中国語/英語)

Phase 1.5: 語義増補
  ├─ ノード間の中間フレームをスキャン
  ├─ Qwen3.5文字認識：ドアプレート、標識、会議室名を検出
  ├─ 品質フィルタリング：ブラックリストで無意味な標識を除外
  ├─ 名前正規化：数字/英語を自動補完(10 → 10号会議室, MOORE → ムーア会議室)
  └─ 空間位置に新ノード挿入 + リナンバリング

Phase 2: 接続生成
  ├─ Qwen3.5 PointGrounderで隣接ノードペアをグラウンディング
  ├─ 廊下中間フレームマッチング：DINOv3 CLS特徴量で最適camera選択
  ├─ ハンガリアンアルゴリズム：camera→隣接ノードの最適割当
  ├─ 3段階crop切り出し(big/mid/small) + Y座標補正
  └─ 出力フォーマット検証 (validate_output.py)
```

### 使用方法

```bash
python auto_mapper/run_auto_map.py \
    --input_dir memory_test_data \
    --output_dir auto_mapper/merged_labeled_data \
    --vpr_config deploy/vpr_config.yaml
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `--input_dir` | `memory_test_data` | 入力画像ディレクトリ |
| `--output_dir` | `auto_mapper/merged_labeled_data` | 出力ディレクトリ |
| `--vpr_config` | `deploy/vpr_config.yaml` | VPR設定ファイル |
| `--similarity_threshold` | `0.70` | VPR類似度閾値 |
| `--min_frame_interval` | `5` | 最小フレーム間隔 |
| `--use_qwen_naming` | `true` | Qwen3.5自動命名 |
| `--qwen_gpu` | `1` | Qwen3.5用GPU |

### 前提条件

1. **vLLMサービス**：`bash deploy/start_qwen_vllm.sh`
2. **VPRモデル**：設定したVPR手法の事前学習重み
3. **DINOv3モデル**：廊下中間フレームマッチング用

### 出力フォーマット

手動アノテーションの`merged_labeled_data/`と完全互換。自動建図データで直接記憶構築：

```bash
bash deploy/build_memory.sh --data_dir auto_mapper/merged_labeled_data
```

### コアコンポーネント

| コンポーネント | 説明 |
|-------------|------|
| `auto_mapper_core.py` | コアコントローラー、3段階Pipelineを編成 |
| `node_distance_estimator.py` | VPR特徴量比較、新ノード作成判定 |
| `auto_landmark_namer.py` | Qwen3.5 vLLMシーン記述 + ランドマーク命名 |
| `semantic_node_detector.py` | ドアプレート/標識文字認識 + 名前正規化 |
| `auto_node_generator.py` | ノードディレクトリ作成、メタデータJSON生成 |
| `auto_sub_image_extractor.py` | PointGrounding + DINOv3廊下フレームマッチング + crop切り出し |
| `validate_output.py` | 出力フォーマット検証 |

---

## 🚀 クイックスタート

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/core_requirements.txt
pip install -e .

# 記憶構築
bash deploy/build_memory.sh

# サービス起動
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
        "front_1": "<base64>",
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
        "phase": "verifying",
        "consecutive_occlusions": 0,
        "occlusion": null,
        "lookahead_conf": 0.68,
        "lookahead_found": true,
        "coord_transform": {
            "yaw_global_deg": -12.3,
            "depression_deg": 8.5,
            "distance": 2.4,
            "elapsed_ms": 0.3
        }
    }
}
```

### 制御コマンド

| コマンド | 説明 |
|----------|------|
| `reset` | Agent + 記憶状態をリセット |
| `toggle_memory` | 記憶ナビのオン/オフ切替 |
| `memory_status` | 記憶ナビ詳細（利用可能な目的地一覧含む） |
| `reset_memory` | 記憶状態のみリセット（Agent履歴保持） |
| `session_status` | セッション状態表示 |

---

## 📋 更新履歴

### v2.2.0

- **🆕 自動建図モジュール**：`auto_mapper/`モジュールを新設、画像シーケンスからトポロジカルグラフを全自動生成
  - 3段階Pipeline：VPRノード作成→語義増補→接続生成
  - Qwen3.5 vLLM推論バックエンド
  - DINOv3廊下中間フレームマッチング + ハンガリアンアルゴリズム
  - 4カメラ並列vLLM呼び出し（全体315s→238s）
  - 出力は`merged_labeled_data/`と完全互換

### v2.1.0 (ドキュメント同期)

- **📝 遮蔽面積閾値修正**：35% → **25%**（コードデフォルトに合致）
- **📝 サブ画像マッチング閾値修正**：0.65 → **0.60**
- **📝 遮蔽トリガー条件修正**：サブ画像マッチング失敗時にトリガー（VPR結果に非依存）
- **📝 側面カメラ回転**：camera_3/camera_4のその場回転ロジックを文書化
- **📝 VPR到達閾値**：`VPR_ARRIVE_THRESHOLD = 0.70` を文書化

### v2.0.0

- **🆕 YOLOv8n遮蔽検出**：`memory_nav/occlusion_detector.py`を新設
  - YOLOv8n（6MB）でperson、backpack等を検出、bbox面積比≥25%で遮蔽判定
  - 遮蔽時は`action: [0, 0, 0]`、解消後に自動でナビゲーション再開
- **🔄 ナビゲーションロジック簡素化**：旧トレンド検出方式を削除
- **🎯 best_fail_camera**：全カメラ失敗時でも最高スコアカメラを記録

### v1.9.0

- **🔭 Lookahead二重確認**：VPR + 次ステップサブ画像マッチングの二重確認
- **🎯 閾値統一**：`SUB_MATCH_CONFIDENCE_THRESHOLD = 0.60`

### v1.8.0

- **🆕 魚眼歪み補正** + **ピクセル→ロボット座標変換** + **cam/ディレクトリ**

### v1.7.0 以前

中文 README の完全な更新履歴をご参照ください。

---

## 📄 ライセンス

本プロジェクトは [MIT License](LICENSE) を採用しています。
