<div align="center">

# 🧠 MemoryNav

**視覚記憶ナビゲーションシステム | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.6.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.6.0)

VPR（視覚的場所認識）とトポロジカルマップに基づくロボット記憶ナビゲーションシステム

[English](README_EN.md) | [中文](README.md) | **日本語** | [한국어](README_KO.md)

</div>

---

## 📖 概要

MemoryNavは移動ロボット向けの視覚記憶ナビゲーションシステムです。4台の全方位魚眼カメラで画像を取得し、VPR技術を用いて事前構築したトポロジカル記憶グラフ上で自己位置推定を行います。YOLOv8nによる視覚遮蔽検出とQwen3.5-9B視覚言語モデルのフォールバックを組み合わせ、「一度行った場所を記憶し、再び歩く」記憶ナビゲーション能力を実現します。

### 主な機能

- **🎯 四分類意図ルーティング**（**v2.5.1**）：Qwen3.5-0.8B vLLM がすべての `task` を `navigate` / `ask_location` / `ask_direction` / `mapping` に自動分類し、それぞれ専用ハンドラーにルーティング。バックエンド優先順位は Qwen3.5-0.8B → Qwen3.5-9B フォールバック → キーワードルールフォールバック
- **📍 現在位置の問い合わせ**：「今どこ？」「現在位置は？」のような質問に対し、VPR のみ実行して `response_text="当前的位置是 X"` を返す。VPR が閾値未満の場合は top-2 類似ノードを用いて「A と B の中間」を返答
- **🧭 道案内**：「X へはどう行く？」のような質問に対し、VPR で現在位置を取得 + `find_destination` で目的地を特定 + 最短経路を計画し、Qwen3.5-0.8B で自然な中国語の道案内文として整形
- **🗺️ 自然言語による在線建図トリガー**：「开始建图」/「启动扫图」/「停止建图」/「完成扫图」のような自然言語発話で `MappingSession` を自動的に開始 / finalize。ハードコードされた `task="mapping"` / `"stop_mapping"` は後方互換（LLM 呼び出しゼロ）
- **🔁 中断からの再開**：ask_location / ask_direction の分岐では `nav_state.plan` / `last_task` を**変更しない**。クライアントが次フレームで `task=None` を送信すれば、保存された状態から元のナビゲーションを継続可能
- **🗺️ 自動建図**：3段階Pipeline（VPRノード作成→語義増補→接続生成）で画像シーケンスからトポロジカルナビゲーショングラフを全自動生成、手動アノテーション不要
- **🔍 マルチスキームVPR測位**：4種類のSOTA VPR手法をサポート、設定ファイル一つで切り替え可能
- **🗺️ トポロジカル記憶グラフ**：アノテーションデータからノード-エッジトポロジーを自動構築、最短経路計画（BFS/Dijkstra）をサポート
- **🔄 循環シフトマッチング**：4カメラ循環シフトアルゴリズムによる方向不変の自己位置推定と偏向角推定
- **🎯 DINOv3サブ画像マッチング**：DINOv3高密度パッチ特徴量による3段階カスケードマッチング（small→mid→big）＋全カメラ走査
- **💾 フレーム間キャッシュ再利用**：DINOv2 VPR特徴量によるフレーム間類似度（追加推論コストゼロ）、マッチング失敗時に前フレームの成功結果をインテリジェントに再利用
- **🔭 Lookahead二重確認**：ステップ切替時にVPR定位と次ステップのサブ画像マッチングを同時検証し、早期advanceを防止
- **📤 統一出力フォーマット**：記憶モードのオン/オフに関わらず一貫したレスポンス形式、常に`pixel_target`を提供
- **🚧 YOLOv8n遮蔽検出**：サブ画像マッチング失敗時にカメラ画面の遮蔽を自動検出（VPR結果とは独立）、遮蔽時はその場で待機し解消後にナビゲーション再開
- **🚦 横断歩道信号機検出** (**v2.6.0**)：from/to ノード名がいずれも「路口」を含むステップ区間で、camera_1/camera_2 に対し YOLO11n + 上下二段 HSV 色判別 + 時系列スムージングで横断歩道信号機を検出。赤信号時はロボットがその場で待機（最優先、サブ画像/Qwen 判断より先に return）、緑信号/無信号は通常ナビゲーション継続。区間を離れると状態を自動クリア
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
│   ├── sub_image_matcher.py        # サブ画像マッチャー (DINOv3高密度特徴量、online_mapper も再利用)
│   ├── occlusion_detector.py       # YOLOv8n遮蔽検出器
│   ├── traffic_light_detector.py   # YOLO11n 信号機検出 (横断歩道区間で起動)
│   ├── fisheye_undistort.py        # 魚眼歪み補正 (円筒投影)
│   ├── coord_transform.py          # ピクセル→ロボット座標変換
│   ├── qwen35_point_grounder.py    # Qwen3.5グラウンディング (フォールバック)
│   ├── qwen35_grounding_server.py  # Qwen3.5サブプロセス推論サーバー
│   ├── vpr_factory.py              # VPR抽出器ファクトリー
│   ├── vpr_config_loader.py        # 設定ローダー
│   └── selavpr_model/              # SelaVPR++モデルコード
├── deploy/                         # デプロイエントリー
│   ├── ws_proxy_with_memory.py     # WebSocketプロキシサービス (メイン、意図分類ルーティング含む)
│   ├── vpr_config.yaml             # VPR統一設定ファイル (selavpr 閾値 0.56)
│   ├── build_memory.sh             # 記憶構築スクリプト
│   ├── start_qwen_vllm.sh          # Qwen3.5-9B vLLM 起動 (GPU 1, port 8199)
│   ├── start_qwen08_vllm.sh        # Qwen3.5-0.8B vLLM 起動 (GPU 0, port 8198)
│   └── start_server.sh             # サーバー起動スクリプト
├── cam/                            # 多眼魚眼カメラ
│   ├── params.yaml                 # カメラパラメータ
│   └── tools/                      # スタンドアロンツール
├── scripts/
│   └── memory_visualization_server.py  # 可視化サービス (サブ画像 + 打点 + 遮蔽検出)
├── pretrained/                     # 事前学習モデル (YOLOv8n, DINOv3等)
├── merged_labeled_data/            # 記憶アノテーションデータ
├── online_mapper/                  # 🛰️ オンライン能動建図モジュール (3 層)
│   ├── run_online_map.py           # CLI エントリ
│   ├── config.py                   # グローバル設定 (depth/vo/occ_backend スイッチ)
│   ├── core/online_mapper_core.py  # ⭐ メインオーケストレーター (process_frame + finalize)
│   ├── geometry/                   # Geometry 層 (VGGT-1B 幾何フロントエンド)
│   │   ├── vggt_backend.py         # ⭐ VGGT-1B シングルトン + スライディングウィンドウ
│   │   ├── depth_estimator.py      #   DA-V2 + VGGTDepthEstimator + ファクトリ
│   │   ├── visual_odometry.py      #   MonoVO + VGGTVisualOdometry + ファクトリ
│   │   ├── pose_graph.py           #   scipy LM ポーズグラフ
│   │   ├── junction_detector.py    #   4 カメラ深度による交差点判定 (stateless)
│   │   ├── traversability.py       # ⭐ VGGT 点群 → ピクセル単位通行可能度 (resolve_crop_point)
│   │   └── occupancy.py            #   1D ray-cast + dense 点群直接充填
│   ├── topology/                   # Topology 層
│   │   ├── keyframe_selector.py    #   多トリガー キーフレーム選択
│   │   ├── loop_closure.py         #   自動閾値 + ORB 幾何検証ループクロージャ
│   │   ├── connection_builder.py   #   ⭐ next_positions: 幾何先験 + traversability + 人物ペナルティ
│   │   ├── graph.py                #   TopoGraph / TopoNode
│   │   └── auto_sub_image_extractor.py  # グラウンディングcrop (memory_nav DINOv3Strategy 再利用)
│   ├── semantics/                  # Semantics 層
│   │   ├── open_set_detector.py    #   Grounding-DINO ラッパー
│   │   ├── door_plate_tracker.py   #   ドアプレート代表フレーム選択
│   │   ├── hallucination_filter.py # ⭐ STRICT プロンプト + QwenVerifier + MultiFrameVoter
│   │   ├── node_category.py        # ⭐ ノードカテゴリー分類器 + CN/EN マップ
│   │   ├── node_naming.py          # ⭐ 構造化命名 NodeName
│   │   ├── colocation_merger.py    # ⭐ 同一位置マージ (NodeName.merge_names を使用)
│   │   ├── scene_graph.py          #   階層的シーングラフ
│   │   └── auto_landmark_namer.py  #   Qwen3.5シーン命名 (vLLM)
│   ├── vpr/
│   │   └── node_distance_estimator.py  # VPRノード距離推定
│   ├── viz/
│   │   └── visualize.py            # pose_graph.png / occupancy.png / keyframe_timeline.png / scene_overview.txt
│   └── io/
│       └── merged_data_writer.py   #   出力ライター + 構造化フィールド
├── third_party/vggt_space/         # VGGT ソース (.gitignore, HF Space からダウンロード)
├── pretrained/                     # モデル重み (.gitignore)
│   ├── vggt-1b/                    #   facebook/VGGT-1B
│   ├── depth-anything-v2-small-hf/ #   バックアップ depth backend
│   ├── grounding-dino-base/        #   IDEA-Research/grounding-dino-base
│   ├── dinov3_vitb16.safetensors   #   VPR バックボーン
│   ├── yolov8n.pt                  #   遮蔽検出
│   └── yolo11n.pt                  #   横断歩道信号機検出
├── tests/
│   └── test_memory_ws.py           # WebSocket統合テスト
└── docs/
    └── online_mapper.md            # 📘 online_mapper 完全設計ドキュメント
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
  ├─ 信号機検出 ("路口→路口" ステップでのみ起動, camera_1+camera_2):
  │   └─ 赤信号 → action=[0,0,0] その場で待機 (最優先, 即 return)
  │      緑信号/none → 後続判断へ
  │      非横断歩道ステップ → reset_state, スキップ
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

## 🎯 四分類意図ルーティング

記憶ナビゲーションに入る前に、サーバーは Qwen3.5-0.8B vLLM を用いてすべてのリクエストを分類し、`task` を 4 つの経路のいずれかにルーティングします:

| 意図 | トリガー例 | ハンドラー | レスポンス形状 |
|------|-----------|-----------|----------------|
| `navigate` | 「C8 前台に行って」「D 棟に案内して」「スタート地点に戻って」 | 記憶ナビゲーションメインフロー | `action=[x,y,yaw]` + `memory_info` |
| `ask_location` | 「今どこ？」「現在位置は？」「現在どこにいる？」 | `handle_ask_location` | `action=[0,0,0]` + `response_text` |
| `ask_direction` | 「D 棟へはどう行く？」「ロビーへの行き方は？」 | `handle_ask_direction` | `action=[0,0,0]` + `response_text` |
| `mapping` | `"mapping"` / `"开始建图"` / `"启动扫图"` / `"停止建图"` / `"完成扫图"` | 建図ライフサイクル（キーワードで開始 / 停止を判定） | `mode="mapping"` + `log` / `mapping` または `summary` |

バックエンド優先順位: **Qwen3.5-0.8B (port 8198)** → Qwen3.5-9B (port 8199) フォールバック → キーワードルールフォールバック。1 回の分類につき約 50 ms。実測 17/17 分類精度（6 個の mapping 自然言語を含む）。

### ask_location レスポンス例

```json
{
  "status": "success",
  "task_status": "executing",
  "action": [[0.0, 0.0, 0.0]],
  "response_text": "当前的位置是微波炉区域",
  "vpr": {
    "matched_node_id": "3",
    "matched_node_name": "微波炉区域",
    "confidence": 0.5629,
    "fallback": null
  },
  "nav_preserved": {
    "has_plan": true,
    "plan_path": ["2", "3", "6", "11"],
    "current_step": 0,
    "total_steps": 3,
    "last_task": "前往C8前台"
  }
}
```

VPR が閾値未満の場合、ハンドラーは top-2 類似ノードを用いて「A と B の中間」として応答します:

```json
{
  "response_text": "目前的位置是c8电梯间和c8前台中间",
  "vpr": {
    "matched_node_id": null,
    "fallback": "between_two_nodes",
    "top1": {"id": "10", "name": "c8电梯间", "sim": 0.3425},
    "top2": {"id": "11", "name": "c8前台",   "sim": 0.2904}
  }
}
```

### ask_direction レスポンス例

```json
{
  "response_text": "您好，您要去 a8 前台，请经过微波炉区域，然后依次经过 c8 打印机、c8 男厕所门口、c8 玻璃门，最后到达实验室门口，再前往 24 号会议室门口即可。",
  "route": {
    "start_name": "微波炉区域",
    "goal_name": "a8前台",
    "total_steps": 5,
    "path_names": ["微波炉区域", "c8打印机", "c8男厕所门口", "c8玻璃门", "实验室门口", "24号会议室门口", "a8前台"]
  },
  "nav_preserved": {"has_plan": true, "current_step": 1, "total_steps": 3, "last_task": "前往C8前台"}
}
```

経路説明文は Qwen3.5-0.8B で整形されます。LLM 失敗時はハンドラーが文字列テンプレートにフォールバックします。

### 中断下でのナビゲーション継続性

ask_location / ask_direction ハンドラーは `nav_state.plan` / `current_step_idx` / `last_task` を**変更しない**ため:

| フレーム | task | 動作 |
|----------|------|------|
| 0 | `"前往C8前台"` | ナビゲーション開始、プラン構築 |
| 1..N-1 | `null` | `last_task` を再利用、継続 |
| K | `"现在在什么位置"` | 現在位置を応答、`nav_state` は変更なし |
| K+1 | `null` | 保存されたステップから継続、phase=verifying |
| M | `"去 X 怎么走"` | 経路を応答、`nav_state` は変更なし |
| M+1 | `null` | 元のプランを継続 |

レスポンス内の `nav_preserved` ブロックにより、UI 側で元のナビゲーションタスクがまだアクティブであることを確認できます。

---

## 🛰️ オンライン能動建図 (online_mapper)

`online_mapper/` は MemoryNav のストリーミング オンライン能動建図モジュールです。VGGT-1B の単一推論で depth / pose / dense 点群を同時取得し、`OnlineMapperCore` が幾何 / トポロジー / 語義の 3 層を編成して高品質なセマンティック トポグラフを生成します。

- **⚙️ 幾何フロントエンド**: VGGT-1B スライディング ウィンドウ 4 フレーム bf16 シングルトン; `VisualOdometry` は VGGT の extrinsics を再利用し追加推論ゼロ; `OccupancyGrid` は dense 点群から直接充填; `Traversability` は点群から地面平面の可通行度を推定し、`resolve_crop_point` で crop 中心を walkable segment 中央に押し付ける (`detect_vertical_obstacle_columns` で柱列マスクを重畳、画面下半分のみスキャンして天井誤判定を回避)
- **🕸️ トポロジー**: 複数トリガー キーフレーム (VPR + 並進 + 回転 + 情報ゲイン + 交差点 + セマンティック ホワイトリスト); 毎フレーム 全域 VPR + ORB 幾何検証ループ閉じ; `ConnectionBuilder` は `next_positions` に幾何方向事前分布 (同 segment ALPHA=0.5 / 300 s 真断絶 ALPHA=0、motion-heading を `pose.theta` より優先, 逆向きはハード ペナルティ) + traversability コリドー補正 + GroundingDINO 人物遮蔽ペナルティ + `cx` エッジ ハード制約を付与; finalize では spatial / temporal KNN で隣接を再構築、cross-gap filter (> 60s かつ bridging keyframe なし → 時間エッジを拒否) 付き; DINOv3 サブ画像マッチングは `memory_nav.sub_image_matcher.DINOv3Strategy` を直接再利用
- **🧠 語義**: マルチカム `describe_scene` 投票 (≥2 cam 一致でノード作成、4 cam 全不一致なら skip) + temporal consensus (FUNCTION_AREA canonical / CROSS / T_JUNCTION は 2/4 で単フレーム免除、それ以外は近 3 フレーム `_recent_scene_winners` に出現していれば verified、LANDMARK_FACILITY は bypass しない); 門牌は STRICT prompt + Qwen 二次検証 + `MultiFrameVoter`, `BUILDING_LANDMARK` は votes ≥ 4 必須 (文字 OCR 幻覚を遮断) かつ数字幻覚の単一フレーム fast-pass は無効化; canonical 正規化 (電梯口 / 電梯間 → 電梯厅; 快递柜 / 外卖柜 / 储物柜 / 智能取餐柜 → 外卖柜区); `_merge_by_canonical_name` で同 base 強制マージ + 3 m 空間クラスタ守衛 + latest-anchor; `NodeName` が構造化命名 `category · organization` を出力; `ColocationMerger` はカテゴリ不一致ガード + 早期 anchor tie-break; 門牌の 2 段階帰属 (functional を先に作成、brand は後 attach、`RELOCATE-DISPLAY` ルールはターゲット ノードの由来に基づき display フレームを再配置するか判断)
- **🎯 出力**: `merged_labeled_data/<id>/node_position_info.json` (構造化 `self_position` + `next_positions` + crops), `scene_graph.json`, `pose_graph.json`, `metrics.json`, `online_mapping_log.jsonl`, `plate_voter_dump.json`
- **🔁 リファレンス実行**: `memory_test_data` 園区巡回 281 フレーム → 8 ノード連鎖 `電梯厅 → 前台 → C座 → H座電梯 → A座 → B座入口 → 外卖柜区·EXHIOH → 2号外卖柜`

### 🌐 WebSocket デュアルモードアクセス

`deploy/ws_proxy_with_memory.py` はポート **9528** で待機し、単一接続で**2 つのモード**をサポートします。すべてのリクエストは同じ形式 `{id, task, pts, images}` を保持し、**4 分類の意図ルーティング** (`navigate / ask_location / ask_direction / mapping`) がモード切替を駆動します:

| `task` の値 | 意図 | 動作 |
|-------------|------|------|
| `"mapping"` / `"开始建图"` / `"启动扫图"` … | `mapping`（開始） | マッピングモードに入る / 継続。初回フレームで `MappingSession` を自動作成し、以降のフレームを `OnlineMapperCore` にフィード |
| `"stop_mapping"` / `"停止建图"` / `"完成扫图"` … | `mapping`（停止） | `finalize` + 可視化をトリガーし、サマリーを返してナビに戻る |
| ナビゲーション / ask-location / ask-direction | `navigate` / `ask_location` / `ask_direction` | 記憶ナビゲーションメインフローを実行。マッピング中だった場合は自動で finalize してからナビに戻る |

制御コマンド (`{"command": "..."}`) は**状態照会用のみ**残されています: `mapping_status` / `memory_status` / `session_status` / `reset` / `reset_memory` / `toggle_memory`。

マッピングモードでは各 `{id, task:"mapping", pts, images: {camera_1..4}}` フレームが `process_mapping_frame` → `OnlineMapperCore.process_frame` を経由し、`finalize` 時にフラッシュされます。SelaVPR 抽出器は `MemoryNavigator.extractor` 経由でナビ・マッピング両モードで**共有**され、二重ロードを回避します。

- 成果物保存先: `deploy/logs/mapping_output/session_{ts}_{client_id}/`（`online_mapper/output/` とは別、後者は `run_online_map.py` ベースライン用）
- 一時フレームディレクトリ: `deploy/logs/mapping_frames/session_*/`、finalize 時に自動クリーンアップ
- クライアント切断時はアクティブなセッションを自動 finalize してデータを保持

---

## 🚀 クイックスタート

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/core_requirements.txt
pip install -e .

# 記憶構築
bash deploy/build_memory.sh
```

### ナビゲーションサービス起動

```bash
# 1. Qwen3.5-9B vLLM 起動 (フォールバックグラウンディング + マッピング命名)
bash deploy/start_qwen_vllm.sh 1 8199

# 2. Qwen3.5-0.8B vLLM 起動 (意図分類 + 経路ナレーション)
bash deploy/start_qwen08_vllm.sh 0 8198

# 3. メインサービス起動 (deploy/vpr_config.yaml を自動読み込み)
python deploy/ws_proxy_with_memory.py
# または: bash deploy/start_server.sh
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

## 🧪 テスト

```bash
python -m pytest tests/unit_test/test_basic.py -v

# ナビゲーション再生 (デフォルト): 初回フレームは完全な TASK を送信、
# 以降のフレームは task=None を送信 (サーバーは last_task を再利用)。
# シーケンスに沿って 3 回の ask_* 中断を均等に注入し、元のナビ状態が
# 保持されることを検証する。
python tests/test_memory_ws.py
python tests/test_memory_ws.py --mode nav

# マッピング再生 — 自動で task="mapping" → 全フレームをフィード → task="stop_mapping"、
# トポロジー / キーフレーム / ドアプレート / 実行時間の内訳 + 成果物パスを出力
python tests/test_memory_ws.py --mode mapping
```

---

## 📋 更新履歴

### v2.6.0

- **🚦 横断歩道信号機検出**: 新規 `memory_nav/traffic_light_detector.py` — YOLO11n 検出 + 上下二段 HSV 色判別 (上赤下緑) + 時系列状態機
  - `from_node` と `to_node` の名称がいずれも「路口」を含むステップでのみ起動 (camera_1 + camera_2 前向き)
  - 赤信号は即座に `action=[0,0,0]` (その場待機) を返却、サブ画像 / Qwen 判断より優先
  - 区間を離れたら `reset_state` で色残留を防止
  - 角度フィルタ (|global_angle| ≤ 30°) + bbox 高フィルタ (≤ 120 px) で側後方/近距離信号を除外
  - レスポンスにトップレベル `traffic_light` フィールドを追加; `memory_info.phase = "red_light"` で停車フレームをマーク
- **🛰️ online_mapper crop パイプライン リファクタ**:
  - 新規 `online_mapper/geometry/traversability.py:resolve_crop_point` で Qwen の crop 中心を**保持せず walkable segment の中心へ押し付け**、obstacle 端で単点 walkable だが crop 半径内に obstacle がある (緑植壁 / 自動改札柵柱) ケースを修正
  - 新規 `detect_vertical_obstacle_columns(bottom_frac=0.5)` の柱列検出は**画面下半分のみ走査**し、2.5 m 天井がすべての列を obstacle と誤判するのを回避
  - 幾何投影フォールバックも traversability で検証 (柱が target cx に投影された場合は Qwen 原点に戻す)
  - ConnectionBuilder 幾何先験 ALPHA を二段階化: 同 segment `0.5` (旧 0.2、幾何先験主導)、300 s 跨ぎ真断絶 `0` (geo_bonus ゼロクリア、純粋に visual sim で判断、VO ドリフト汚染を防止)
  - DINOv3 サブ画像マッチングは `memory_nav.sub_image_matcher.DINOv3Strategy` を直接再利用、コード重複を解消
- **🛰️ online_mapper ノードマージ改善**:
  - `_merge_by_canonical_name` に 3 m 空間クラスタ守衛を追加: 同 canonical 名でも欧氏距離 > 3 m なら非マージ (起点エレベーターホール vs H 棟横エレベーターが両方「電梯厅」になり誤マージされる問題を修正)
  - anchor を最新 `frame_idx` に変更 (経路上で landmark を通過した後の keyframe ほど landmark に近く plate bbox も大きい)
- **🛰️ online_mapper temporal consensus 厳格化**:
  - LANDMARK_FACILITY の単フレーム ホワイトリスト fast-pass を削除、N11 電梯厅\_2 タイプの多カメラ共同幻覚を遮断
  - CROSS / T_JUNCTION 場面は 2/4 consensus で temporal を免除 (前台/関爱室類 landmark の単フレーム認識を救済)
  - FUNCTION_AREA canonical winner は 2/4 consensus で temporal を免除 (FUNCTION_AREA は Qwen 低幻覚の具体 landmark)
- **🛰️ online_mapper plate 投票**:
  - BUILDING_LANDMARK plate は votes ≥ 4 で confirm、Qwen OCR 字母幻覚を遮断 (`13号楼` / `D座` 等の幻覚は通常 votes ≤ 3、最小実例 `B座` = 4)
  - BUILDING_LANDMARK の単フレーム ホワイトリスト fast-pass を無効化

### v2.5.1

- **🗺️ オンライン建図を第 4 の意図クラスに昇格**: 意図分類が 3 → 4 クラスへ (navigate / ask_location / ask_direction / **mapping**)
  - 自然言語「开始建图」/「启动扫图」/「请开始建图」等が自動で `MappingSession` を作成
  - 自然言語「停止建图」/「结束建图」/「完成扫图」等が自動で `finalize` をトリガー
  - ハードコード `task="mapping"` / `"stop_mapping"` は後方互換（LLM 呼び出しゼロ）
  - 4 クラス意図分類 17/17 精度を実測
- **🔧 `handle_client` での統一ディスパッチ**: 意図分類を `process_inference_with_memory` から `handle_client` 最上層に引き上げ、intent 値で直接ルーティング; `process_inference_with_memory` に `intent` パラメータを追加して重複分類を回避

### v2.5.0

- **🎯 意図分類ルーティング**: Qwen3.5-0.8B vLLM による新 `IntentClassifier`。すべての `task` を navigate / ask_location / ask_direction に自動分類、1 回あたり約 50 ms
  - バックエンド優先順位: Qwen3.5-0.8B (8198) → Qwen3.5-9B (8199) フォールバック → キーワードルールフォールバック
  - 新ランチャー `deploy/start_qwen08_vllm.sh` (GPU 0, メモリ使用量約 4.6 GB)
- **📍 現在位置の問い合わせ (`handle_ask_location`)**: 「今どこ？」「現在位置は？」のような質問に対し VPR のみ実行し、`response_text="当前的位置是 X"` を返却。VPR 類似度が低い場合は top-2 最近接ノードを用いて「A と B の中間」を返答
- **🧭 道案内 (`handle_ask_direction`)**: VPR で現在位置取得 + `find_destination` で目的地特定 + `plan_navigation` で経路計画後、Qwen3.5-0.8B が経路を自然な中国語文として整形 (テンプレートフォールバックあり)
- **🔁 ナビゲーション継続性**: ask_location / ask_direction は `nav_state` / `session_state['last_task']` を一切変更しない。クライアントは次フレームで `task=None` を送信するだけで元のプランをシームレスに再開可能
- **🎛️ 閾値チューニング**: VPR `similarity_threshold.selavpr` 0.60 → **0.56**、`VPR_ARRIVE_THRESHOLD` 0.70 → **0.68**。テストセットで 48/49 フレーム VPR ヒット + 初めて完全なナビゲーション完了を達成
- **🧪 test_memory_ws.py 強化**: 初回フレームで完全な TASK を送信、以降のフレームは `task=None` (サーバーが `last_task` を再利用)。3 回の ask_* 中断をシーケンスに沿って均等に注入し、`nav_preserved` が 100% 維持されることを検証
- **🐛 バグ修正**: `process_inference_with_memory` 内のローカル `import math` がモジュールレベルのインポートをシャドーイングしていた問題を修正

### v2.3.0

- **🛰️ オンライン能動建図モジュール** (`online_mapper/`): 3 層アーキテクチャ (Geometry + Topology + Semantics) によるストリーミング オンライン建図モジュール
  - **Geometry 層**: 単眼 ORB+EssentialMatrix VO、Depth-Anything-V2、scipy LM ポーズグラフ、2D 占有グリッド、4 カメラ深度交差点検出
  - **Topology 層**: マルチトリガー キーフレーム、自動閾値 + ORB 幾何検証によるグローバルループクロージャ、空間 KNN ∪ 時間隣接による隣接関係再構築
  - **Semantics 層**: STRICT プロンプト + QwenVerifier 二次検証 + MultiFrameVoter 多フレーム投票 + サブストリング変異マージ + 7 カテゴリーホワイトリスト + ColocationMerger 同一位置マージ + CN/EN バイリンガル命名 + NameDeduplicator サフィックス重複解消
  - `merged_labeled_data/` スキーマを生成、追加で `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` を出力
  - **テストデータ (49 フレーム) 最終結果**: 5 ノード (印刷エリア / 受付 / NEUMANN 電気室 / ケアルーム / DEEPROUTE.AI 受付)、幻覚 0 / 重複 0 / ループクロージャ 2 回、validator 5/5 合格
  - 完全設計ドキュメント: **[`docs/online_mapper.md`](docs/online_mapper.md)**
  - イテレーション履歴 (r1→r6): [`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)

### v2.2.0

- **🆕 自動建図モジュール**: 画像シーケンスからトポロジカルグラフを全自動生成
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
