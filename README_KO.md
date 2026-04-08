<div align="center">

# 🧠 MemoryNav

**시각적 기억 내비게이션 시스템 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.2.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.2.0)

VPR(시각적 장소 인식)과 위상 지도 기반 로봇 기억 내비게이션 시스템

[English](README_EN.md) | [中文](README.md) | [日本語](README_JA.md) | **한국어**

</div>

---

## 📖 소개

MemoryNav는 이동 로봇을 위한 시각적 기억 내비게이션 시스템입니다. 4개의 전방위 어안 카메라로 이미지를 수집하고, VPR 기술을 사용하여 사전 구축된 위상 기억 그래프에서 위치를 추정하며, YOLOv8n 시각적 차폐 감지와 Qwen3.5-9B 시각-언어 모델 폴백을 결합하여 "한번 간 곳을 기억하고 다시 걷는" 기억 내비게이션 능력을 실현합니다.

### 주요 기능

- **🗺️ 자동 맵 생성**：3단계 Pipeline(VPR 노드 생성 → 의미 보강 → 연결 생성)으로 이미지 시퀀스에서 위상 내비게이션 그래프를 자동 생성, 수동 어노테이션 불필요
- **🔍 멀티 스킴 VPR 측위**：4가지 SOTA VPR 방법 지원, 설정 파일 하나로 전환 가능
- **🗺️ 위상 기억 그래프**：어노테이션 데이터에서 노드-엣지 위상을 자동 구축, BFS/Dijkstra 최단 경로 계획 지원
- **🔄 순환 시프트 매칭**：4카메라 순환 시프트 알고리즘으로 방향 불변 위치 추정 및 편향각 추정
- **🎯 DINOv3 서브 이미지 매칭**：DINOv3 고밀도 패치 특징의 3단계 캐스케이드 매칭(small→mid→big) + 전체 카메라 스캔
- **💾 프레임 간 캐시 재사용**：DINOv2 VPR 특징 기반 프레임 간 유사도(추가 추론 비용 제로), 매칭 실패 시 이전 프레임 성공 결과를 지능적으로 재사용
- **🔭 Lookahead 이중 확인**: 단계 전환 시 VPR 위치 확인과 다음 단계 서브이미지 매칭을 동시 검증하여 조기 advance 방지
- **📤 통합 출력 형식**：기억 모드 온/오프에 관계없이 일관된 응답 형식, 항상 `pixel_target` 제공
- **🚧 YOLOv8n 차폐 감지**：서브 이미지 매칭 실패 시 카메라 화면의 차폐를 자동 감지(VPR 결과와 독립), 차폐 시 정지 대기 후 해소되면 내비게이션 재개
- **🤖 Qwen3.5 폴백 그라운딩**：2단계 추론(존재 확인 + 조건부 그라운딩)으로 Qwen3.5-9B VLM 폴백
- **📷 어안 왜곡 보정**：시작 시 `cam/params.yaml`에서 카메라 내부 파라미터 로드, 원통형 투영 왜곡 보정 적용
- **🧭 픽셀→로봇 좌표 변환**：정규화된 `pixel_target`을 완전 물리 파이프라인으로 로봇 운동 좌표로 변환
- **🔄 측면 카메라 회전 처리**：camera_3/camera_4 매칭 성공 시 목표 방향으로의 회전 동작 자동 출력
- **🌐 WebSocket 서비스**：실시간 스트리밍으로 이미지 수신 및 내비게이션 명령 반환
- **⚙️ 통합 설정 관리**：모든 VPR 파라미터를 `deploy/vpr_config.yaml`에 집중 관리

---

## 🏗️ 시스템 아키텍처

```
MemoryNav/
├── memory_nav/                     # 핵심 메모리 내비게이션 모듈
│   ├── memory_navigator.py         # 내비게이터 메인 인터페이스
│   ├── memory_models.py            # 데이터 모델 (Node, Edge, Plan, VPRResult)
│   ├── memory_graph.py             # 토폴로지 그래프 (BFS/Dijkstra)
│   ├── memory_vpr.py               # VPR 매칭 엔진 (순환 시프트 + 순서 불변)
│   ├── memory_builder.py           # 메모리 빌더
│   ├── sub_image_matcher.py        # 서브 이미지 매처 (DINOv3 밀집 특징)
│   ├── occlusion_detector.py       # YOLOv8n 차폐 검출기
│   ├── fisheye_undistort.py        # 어안 왜곡 보정 (원통형 투영)
│   ├── coord_transform.py          # 픽셀→로봇 좌표 변환
│   ├── qwen35_point_grounder.py    # Qwen3.5 그라운딩 (폴백)
│   ├── qwen35_grounding_server.py  # Qwen3.5 서브프로세스 추론 서버
│   ├── vpr_factory.py              # VPR 추출기 팩토리
│   ├── vpr_config_loader.py        # 통합 설정 로더
│   └── selavpr_model/              # SelaVPR++ 모델 코드
├── deploy/                         # 배포 엔트리
│   ├── ws_proxy_with_memory.py     # WebSocket 프록시 서비스 (메인 엔트리)
│   ├── vpr_config.yaml             # VPR 통합 설정 파일
│   ├── build_memory.sh             # 메모리 구축 스크립트
│   └── start_server.sh             # 서버 시작 스크립트
├── cam/                            # 다안 어안 카메라
│   ├── params.yaml                 # 카메라 파라미터
│   └── tools/                      # 독립 도구
├── scripts/
│   └── memory_visualization_server.py  # 시각화 서비스 (서브 이미지 + 포인팅 + 차폐 감지)
├── pretrained/                     # 사전 학습 모델 (YOLOv8n, DINOv3 등)
├── merged_labeled_data/            # 메모리 어노테이션 데이터
├── offline_mapper/                 # 🗺️ 오프라인 맵 생성 모듈 (구 auto_mapper)
│   ├── run_auto_map.py             # 엔트리 스크립트
│   ├── auto_mapper_core.py         # 코어 컨트롤러 (3단계 Pipeline)
│   ├── node_distance_estimator.py  # VPR 노드 거리 추정
│   ├── auto_landmark_namer.py      # Qwen3.5 장면 명명 (vLLM)
│   ├── semantic_node_detector.py   # 도어플레이트/표지판 문자 인식
│   ├── auto_node_generator.py      # 노드 디렉토리 및 메타데이터 생성
│   ├── auto_sub_image_extractor.py # 그라운딩 crop + 복도 프레임 매칭
│   └── validate_output.py          # 출력 형식 검증
├── online_mapper/                  # 🛰️ 온라인 능동 맵 생성 모듈 (v2.3.0, 3 계층)
│   ├── run_online_map.py           # CLI 엔트리
│   ├── config.py                   # 글로벌 설정 (depth/vo/occ_backend 스위치)
│   ├── core/online_mapper_core.py  # ⭐ 메인 오케스트레이터 (~870 줄)
│   ├── geometry/                   # Geometry 계층 (VGGT-1B 기하 프론트엔드)
│   │   ├── vggt_backend.py         # ⭐ VGGT-1B 싱글톤 + 슬라이딩 윈도우 (NEW v2.2)
│   │   ├── depth_estimator.py      #   DA-V2 + VGGTDepthEstimator + 팩토리
│   │   ├── visual_odometry.py      #   MonoVO + VGGTVisualOdometry + 팩토리
│   │   ├── pose_graph.py           #   scipy LM 포즈 그래프
│   │   ├── junction_detector.py    #   4 카메라 깊이 교차로 (stateless)
│   │   └── occupancy.py            #   1D ray-cast + dense 점군 직접 채움
│   ├── topology/                   # Topology 계층
│   │   ├── keyframe_selector.py    #   다중 트리거 키프레임 선택
│   │   ├── loop_closure.py         #   자동 임계값 + ORB 기하 검증
│   │   ├── connection_builder.py   #   ⭐ next_positions: 기하 방향 사전
│   │   └── graph.py                #   TopoGraph / TopoNode
│   ├── semantics/                  # Semantics 계층
│   │   ├── open_set_detector.py    #   Grounding-DINO 래퍼
│   │   ├── door_plate_tracker.py   #   도어플레이트 대표 프레임 선택
│   │   ├── hallucination_filter.py # ⭐ STRICT 프롬프트 + QwenVerifier + MultiFrameVoter
│   │   ├── node_category.py        # ⭐ 노드 카테고리 분류기 + CN/EN 매핑
│   │   ├── node_naming.py          # ⭐ 구조화 명명 NodeName (NEW v2.3)
│   │   ├── colocation_merger.py    # ⭐ 동일 위치 병합 (NodeName.merge_names 사용)
│   │   └── scene_graph.py          #   계층적 씬 그래프
│   └── io/
│       └── merged_data_writer.py   #   출력 라이터 + 구조화 필드
├── third_party/vggt_space/         # VGGT 소스 (.gitignore, HF Space에서 다운로드)
├── pretrained/                     # 모델 가중치 (.gitignore)
│   ├── vggt-1b/                    #   facebook/VGGT-1B
│   ├── depth-anything-v2-small-hf/ #   백업 depth backend
│   ├── grounding-dino-base/        #   IDEA-Research/grounding-dino-base
│   └── dinov3_vitb16.safetensors   #   VPR 백본
├── tests/
│   └── test_memory_ws.py           # WebSocket 통합 테스트
└── docs/
    └── online_mapper.md            # 📘 online_mapper 전체 설계 문서 (v2.3.0)
```

---

## 📷 어안 왜곡 보정

VPR 매칭 및 서브 이미지 매칭 전에 4채널 어안 이미지에 대해 자동으로 원통형 투영 왜곡 보정을 수행합니다.

1. 시작 시 `cam/params.yaml`에서 각 카메라의 내부 파라미터(`xi, fx, fy, cx, cy`)와 왜곡 계수(`k1, k2, p1, p2`) 로드
2. 카메라당 한 번 원통형 투영 remap 테이블 사전 계산(`pitch_up` 오프셋 포함)
3. 각 추론 프레임 전 `cv2.remap` 적용, 계산 비용 극히 낮음
4. `cam/params.yaml` 없을 경우 왜곡 보정 스킵, 서비스 정상 동작 유지

---

## 🧭 픽셀→로봇 좌표 변환

정규화된 `pixel_target: [x_norm, y_norm]`을 전체 물리 파이프라인을 통해 로봇 운동 좌표 `[x_forward, y_lateral, 0.0]`으로 변환합니다.

### 측면 카메라 회전 처리

camera_3 또는 camera_4(후방 향)가 매칭에 성공한 경우, 전진 동작 대신 좌표 변환으로 실제 yaw 각도를 계산하여 제자리 회전 동작 `[0, 0, yaw_rad]`을 출력합니다.

### 카메라 방위각 (`cam/params.yaml` T_ic에서 산출)

| 카메라 | 방위각 |
|--------|--------|
| camera_1 | +39.42° |
| camera_2 | −35.84° |
| camera_3 | −142.04° |
| camera_4 | +143.52° |

---

## 🎯 서브 이미지 매칭 내비게이션

**DINOv3** 고밀도 패치 특징을 이용한 서브 이미지 매칭：

1. **기억 구축 시**：각 엣지에 `camera_name`과 3단계 `crop_image`(big/mid/small) 어노테이션
2. **내비게이션 실행 시**：전체 4카메라를 small→mid→big 3단계 캐스케이드로 스캔, 글로벌 최고 conf 결과 선택
3. **목표 측위**：DINOv3 ViT-B/16으로 고밀도 패치 토큰 추출 → 슬라이딩 윈도우 + unfold 가속 → 코사인 유사도 최대 위치 → `pixel_target`으로 출력
4. **매칭 임계값**：신뢰도 ≥ **0.60** (`SUB_MATCH_CONFIDENCE_THRESHOLD`) 시 성공
5. **프레임 간 캐시**：낮은 신뢰도 프레임에서 DINOv2 VPR 특징 유사도(임계값 **0.70**)로 이전 프레임 결과 재사용 판단, 스텝 전환 시 초기화
6. **폴백**：캐시 없을 경우 Qwen3.5 포인팅 그라운딩 트리거

---

## 🚧 차폐 감지

서브 이미지 매칭 실패 시(VPR 결과와 독립), 주의 카메라에 대해 차폐 감지를 자동 실행：

1. **트리거**：서브 이미지 매칭 실패 시 즉시 트리거 — VPR 결과에 비의존
2. **카메라 선택**：서브 이미지 매칭 점수가 가장 높은(임계값 미만) camera 사용
3. **YOLOv8n 추론**：근거리 전경 물체(person, backpack, umbrella, handbag, suitcase) 감지, bbox 면적비 계산
4. **차폐 판정**：단일 차폐물 면적비 ≥ **25%**(기본값) → 차폐 판정
5. **차폐 시**：`action: [0, 0, 0]`(정지 대기), 서브 이미지 캐시 초기화
6. **차폐 없을 경우**：Qwen3.5 포인팅(고정 "통로 중앙+깊이" 경로 탐색)으로 폴백 내비게이션

### 내비게이션 결정 흐름

```
프레임당:
  ├─ 서브 이미지 매칭 (전체 4카메라 × 3단계 cascade)
  ├─ Lookahead 다음 스텝 서브 이미지 매칭
  │
  ├─ 서브 이미지 매칭 실패 시:
  │   ├─ YOLOv8n 차폐 감지 (최고 점수 camera에서)
  │   │   ├─ 차폐 → action=[0,0,0] 대기, 캐시 초기화
  │   │   └─ 차폐 없음 → Qwen3.5 포인팅 → 실패 시 기억 재전송
  │
  ├─ VPR 성공:
  │   ├─ 목표 노드 + sim≥0.70:
  │   │   ├─ 마지막 스텝 → 직접 advance
  │   │   ├─ 다음 스텝 매칭 OK → Lookahead 확인 → advance
  │   │   └─ 다음 스텝 매칭 NG → VPR HELD
  │   └─ 기타 노드 / sim<0.70 → 현재 스텝 계속
  │
  └─ VPR 실패:
      ├─ 서브매칭 OK → 서브매칭 결과로 내비 계속
      ├─ 서브매칭 NG + Qwen3.5 OK → 포인팅 결과로 내비
      └─ 전부 실패 → 기억 안내 재전송
```

---

## ✨ VPR 방법 비교

| 방법 | 키 | 발표 | 특징 차원 | 백본 | 특징 |
|------|-----|------|---------|------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **권장**, 해싱+리랭킹 |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | 종합 최강, 다중 데이터셋 SOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 다층 CLS | 경량·고속 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 가변 | DINOv2-B + VLAD | 클래식, 안정적 |

---

## 🗂️ 맵 생성 모듈 비교

MemoryNav는 서로 **상호 보완**하는 두 개의 맵 생성 모듈을 제공합니다:

| 차원 | `offline_mapper/` (오프라인) | `online_mapper/` (온라인 능동) |
|---|---|---|
| **포지셔닝** | 녹화 완료 후 일괄 후처리 | 로봇 주행 중 스트리밍 의사결정 |
| **시간적 가정** | 전체 프레임 가시 | "도달한" 프레임만 |
| **메인 루프** | 3 단계 Pipeline (Phase1 → 1.5 → 2) | 프레임별: geometry → VPR → 루프 → 플레이트 스캔 → KF 트리거 → 분류 → 노드 생성 |
| **키프레임 전략** | VPR 유사도 + 최소 프레임 간격 | VPR + 누적 병진 + 누적 회전 + 정보 이득 + 교차로 + 시맨틱 화이트리스트 |
| **루프 클로저** | 시작-끝 비교 (선택적) | 전역 VPR + ORB 기하 검증, 매 프레임 |
| **명명** | Qwen describe_scene / detect_text (단일 프레임) | 다중 프레임 투표 + 2차 검증 + 카테고리 화이트리스트 + CN/EN 이중 언어 |
| **노드 필터** | 없음 (모든 VPR 트리거가 노드 생성) | 7 카테고리 화이트리스트, 장식 벽/화분/빈 복도 거부 |
| **환각 방어** | 없음 | STRICT 프롬프트 + QwenVerifier + MultiFrameVoter + 부분 문자열 변이 병합 |
| **출력 스키마** | `merged_labeled_data/` | **완전 호환** `merged_labeled_data/` + 추가 `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` |
| **코드 관계** | 독립형 | `offline_mapper.AutoSubImageExtractor` / `AutoLandmarkNamer` / `NodeDistanceEstimator` 서브클래싱 및 재사용 (offline_mapper 수정 없음) |

두 모듈 모두 **동일한 출력 스키마**를 가지며 `deploy/build_memory.sh`로 직접 메모리 구축에 사용할 수 있습니다.

전체 online_mapper 설계 문서: **[`docs/online_mapper.md`](docs/online_mapper.md)** (13 장, 약 47k 문자)
online_mapper 이터레이션 기록 (r1→r6): **[`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)**

---

## 🗺️ 오프라인 맵 생성 (offline_mapper)

오프라인 맵 생성 모듈(`offline_mapper/`, 구 `auto_mapper`)은 로봇이 촬영한 이미지 시퀀스에서 위상 내비게이션 그래프를 자동으로 생성합니다. 수동 어노테이션이 필요 없습니다.

### 3단계 Pipeline

```
Phase 1: VPR 조립도 노드 생성
  ├─ 프레임 순서로 4카메라 이미지 스캔
  ├─ VPR 특징 추출 → 가장 가까운 노드와 유사도 비교
  ├─ 유사도 < 임계값(0.70) → 새 노드 생성
  └─ Qwen3.5 VLM 자동 명명(중/영문)

Phase 1.5: 의미 보강
  ├─ 노드 간 중간 프레임 스캔
  ├─ Qwen3.5 문자 인식: 도어플레이트, 표지판, 회의실 이름 감지
  ├─ 품질 필터링: 블랙리스트로 무의미한 표지판 제외
  ├─ 이름 정규화: 숫자/영문 자동 보완
  └─ 공간 위치에 새 노드 삽입 + 번호 재할당

Phase 2: 연결 생성
  ├─ Qwen3.5 PointGrounder로 인접 노드 쌍 그라운딩
  ├─ 복도 중간 프레임 매칭: DINOv3 CLS 특징으로 최적 camera 선택
  ├─ 헝가리안 알고리즘: camera → 이웃 노드 최적 할당
  ├─ 3단계 crop 추출(big/mid/small) + Y좌표 보정
  └─ 출력 형식 검증 (validate_output.py)
```

### 사용 방법

```bash
python offline_mapper/run_auto_map.py \
    --input_dir memory_test_data \
    --output_dir offline_mapper/merged_labeled_data \
    --vpr_config deploy/vpr_config.yaml
```

### 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--input_dir` | `memory_test_data` | 입력 이미지 디렉토리 |
| `--output_dir` | `offline_mapper/merged_labeled_data` | 출력 디렉토리 |
| `--vpr_config` | `deploy/vpr_config.yaml` | VPR 설정 파일 |
| `--similarity_threshold` | `0.70` | VPR 유사도 임계값 |
| `--min_frame_interval` | `5` | 최소 프레임 간격 |
| `--use_qwen_naming` | `true` | Qwen3.5 자동 명명 |
| `--qwen_gpu` | `1` | Qwen3.5용 GPU |

### 전제 조건

1. **vLLM 서비스**：`bash deploy/start_qwen_vllm.sh`
2. **VPR 모델**：설정된 VPR 방법의 사전 학습 가중치
3. **DINOv3 모델**：복도 중간 프레임 매칭용

### 출력 형식

수동 어노테이션 `merged_labeled_data/`와 완전 호환。자동 생성 데이터로 바로 메모리 구축：

```bash
bash deploy/build_memory.sh --data_dir offline_mapper/merged_labeled_data
```

### 코어 컴포넌트

| 컴포넌트 | 설명 |
|---------|------|
| `offline_mapper/auto_mapper_core.py` | 코어 컨트롤러, 3단계 Pipeline 편성 |
| `node_distance_estimator.py` | VPR 특징 비교, 새 노드 생성 판정 |
| `auto_landmark_namer.py` | Qwen3.5 vLLM 장면 기술 + 랜드마크 명명 |
| `semantic_node_detector.py` | 도어플레이트/표지판 문자 인식 + 이름 정규화 |
| `auto_node_generator.py` | 노드 디렉토리 생성, 메타데이터 JSON 생성 |
| `auto_sub_image_extractor.py` | PointGrounding + DINOv3 복도 프레임 매칭 + crop 추출 |
| `validate_output.py` | 출력 형식 검증 |

---

## 🚀 빠른 시작

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/core_requirements.txt
pip install -e .

# 기억 구축
bash deploy/build_memory.sh

# 서비스 시작
python deploy/ws_proxy_with_memory.py
```

---

## 📡 WebSocket 프로토콜

### 요청

```json
{
    "id": "robot_01",
    "pts": 1709558400,
    "task": "안내 데스크로 이동",
    "images": {
        "front_1": "<base64>",
        "camera_1": "<base64>",
        "camera_2": "<base64>",
        "camera_3": "<base64>",
        "camera_4": "<base64>"
    }
}
```

### 응답

```json
{
    "status": "success",
    "id": "robot_01",
    "task_status": "executing",
    "action": [[0.5, -0.1, 0.0]],
    "pixel_target": [0.485, 0.521],
    "memory_active": true,
    "camera_name": "camera_2",
    "landmark_name": "엘리베이터",
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

### 제어 명령

| 명령 | 설명 |
|------|------|
| `reset` | Agent + 기억 상태 초기화 |
| `toggle_memory` | 기억 내비 온/오프 전환 |
| `memory_status` | 기억 내비 상세(사용 가능한 목적지 목록 포함) |
| `reset_memory` | 기억 상태만 초기화(Agent 이력 유지) |
| `session_status` | 세션 상태 표시 |

---

## 📋 업데이트 로그

### v2.3.0

- **🛰️ 온라인 능동 맵 생성 모듈** (`online_mapper/`): 3 계층 아키텍처 (Geometry + Topology + Semantics) 기반의 스트리밍 온라인 맵 생성 모듈 신설, `offline_mapper/`와 상호 보완
  - **Geometry 계층**: 단안 ORB+EssentialMatrix VO, Depth-Anything-V2, scipy LM 포즈 그래프, 2D 점유 그리드, 4-카메라 깊이 교차로 감지
  - **Topology 계층**: 다중 트리거 키프레임, 자동 임계값 + ORB 기하 검증 전역 루프 클로저, 공간 KNN ∪ 시간 인접 인접성 재구축
  - **Semantics 계층**: STRICT 프롬프트 + QwenVerifier 2차 검증 + MultiFrameVoter 다중 프레임 투표 + 부분 문자열 변이 병합 + 7 카테고리 화이트리스트 + ColocationMerger 동일 위치 병합 + CN/EN 이중 언어 명명 + NameDeduplicator 접미사 중복 해결
  - 출력 스키마는 `offline_mapper/`와 100% 호환, 추가로 `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` 생성
  - **테스트 데이터 (49 프레임) 최종 결과**: 고품질 노드 5 개 (인쇄 구역 / 리셉션 / NEUMANN 전기실 / 케어 룸 / DEEPROUTE.AI 리셉션), 환각 0 / 중복 0 / 루프 클로저 2 회, validator 5/5 통과
  - 전체 설계 문서: **[`docs/online_mapper.md`](docs/online_mapper.md)**
  - 이터레이션 기록 (r1→r6): [`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)
- **🔁 `auto_mapper/` → `offline_mapper/` 이름 변경**: `online_mapper/`와 짝을 맞춰 오프라인/온라인 맵 생성을 명확히 구분. 내부 클래스 이름 (`AutoMapperCore` 등)은 의도적으로 유지하고 import 경로만 이동.

### v2.2.0

- **🆕 자동 맵 생성 모듈**：`auto_mapper/` 모듈 (v2.3.0에서 `offline_mapper/`로 이름 변경) 신설, 이미지 시퀀스에서 위상 그래프를 자동 생성
  - 3단계 Pipeline：VPR 노드 생성 → 의미 보강 → 연결 생성
  - Qwen3.5 vLLM 추론 백엔드
  - DINOv3 복도 중간 프레임 매칭 + 헝가리안 알고리즘
  - 4카메라 병렬 vLLM 호출(전체 315s→238s)
  - 출력은 `merged_labeled_data/`와 완전 호환

### v2.1.0 (문서 동기화)

- **📝 차폐 면적 임계값 수정**：35% → **25%**(코드 기본값에 일치)
- **📝 서브 이미지 매칭 임계값 수정**：0.65 → **0.60**
- **📝 차폐 트리거 조건 수정**：서브 이미지 매칭 실패 시 트리거(VPR 결과에 비의존)
- **📝 측면 카메라 회전**：camera_3/camera_4의 제자리 회전 로직 문서화
- **📝 VPR 도착 임계값**：`VPR_ARRIVE_THRESHOLD = 0.70` 문서화

### v2.0.0

- **🆕 YOLOv8n 차폐 감지**：`memory_nav/occlusion_detector.py` 신설
  - YOLOv8n(6MB)으로 person, backpack 등 감지, bbox 면적비 ≥ 25%로 차폐 판정
  - 차폐 시 `action: [0, 0, 0]`, 해소 후 자동 내비게이션 재개
- **🔄 내비게이션 로직 간소화**：기존 트렌드 감지 방식 삭제
- **🎯 best_fail_camera**：전체 실패 시에도 최고 점수 카메라 기록

### v1.9.0

- **🔭 Lookahead 이중 확인**: VPR + 다음 단계 서브이미지 매칭 이중 확인
- **🎯 임계값 통합**: `SUB_MATCH_CONFIDENCE_THRESHOLD = 0.60`

### v1.8.0

- **🆕 어안 왜곡 보정** + **픽셀→로봇 좌표 변환** + **cam/ 디렉토리**

### v1.7.0 이전

중문 README의 전체 업데이트 이력을 참조하세요.

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 사용합니다.
