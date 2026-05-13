<div align="center">

# 🧠 memory-nav

**시각적 기억 내비게이션 시스템 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.6.0-orange.svg)](https://github.com/jx1100370217/memory-nav/releases/tag/v2.6.0)

VPR(시각적 장소 인식)과 위상 지도 기반 로봇 기억 내비게이션 시스템

[English](README_EN.md) | [中文](README.md) | [日本語](README_JA.md) | **한국어**

</div>

---

## 📖 소개

memory-nav는 이동 로봇을 위한 시각적 기억 내비게이션 시스템입니다. 4개의 전방위 어안 카메라로 이미지를 수집하고, VPR 기술을 사용하여 사전 구축된 위상 기억 그래프에서 위치를 추정하며, YOLOv8n 시각적 차폐 감지와 Qwen3.5-9B 시각-언어 모델 폴백을 결합하여 "한번 간 곳을 기억하고 다시 걷는" 기억 내비게이션 능력을 실현합니다.

### 주요 기능

- **🎯 4 클래스 의도 라우팅** (**v2.5.1**)：Qwen3.5-0.8B vLLM 이 모든 `task` 를 navigate / ask_location / ask_direction / mapping 로 자동 분류하고 각 핸들러로 라우팅. 백엔드 우선순위 Qwen3.5-0.8B → Qwen3.5-9B 폴백 → 키워드 룰 폴백
- **📍 현재 위치 질의**："현재 위치"·"내가 어디에 있나" 질의에 VPR 만 실행하여 `response_text="当前的位置是 X"` 반환. VPR 임계값 미달 시 유사도 top-2 노드를 "A 와 B 사이" 로 응답
- **🧭 길 안내 질의**："X 로 어떻게 가나"·"로비로 가는 길" 질의에 VPR 로 출발지 + `find_destination` 으로 목적지 + 최단 경로 계획 후 Qwen3.5-0.8B 가 자연스러운 중국어 경로 안내를 생성
- **🗺️ 자연어로 온라인 맵핑 트리거**："开始建图" / "启动扫图" / "停止建图" / "完成扫图" 같은 자연어 발화가 자동으로 `MappingSession` 을 시작 / 종료. 하드코딩된 `task="mapping"` / `"stop_mapping"` 은 하위 호환 (LLM 호출 없음)
- **🔁 중단 후 재개**：ask_location / ask_direction 브랜치는 `nav_state.plan` / `last_task` 를 변경하지 않으므로, 다음 프레임에서 `task=None` 만 보내면 서버가 보존된 상태로 원래 내비게이션을 계속 수행
- **🗺️ 자동 맵 생성**：3단계 Pipeline(VPR 노드 생성 → 의미 보강 → 연결 생성)으로 이미지 시퀀스에서 위상 내비게이션 그래프를 자동 생성, 수동 어노테이션 불필요
- **🔍 멀티 스킴 VPR 측위**：4가지 SOTA VPR 방법 지원, 설정 파일 하나로 전환 가능
- **🗺️ 위상 기억 그래프**：어노테이션 데이터에서 노드-엣지 위상을 자동 구축, BFS/Dijkstra 최단 경로 계획 지원
- **🔄 순환 시프트 매칭**：4카메라 순환 시프트 알고리즘으로 방향 불변 위치 추정 및 편향각 추정
- **🎯 DINOv3 서브 이미지 매칭**：DINOv3 고밀도 패치 특징의 3단계 캐스케이드 매칭(small→mid→big) + 전체 카메라 스캔
- **💾 프레임 간 캐시 재사용**：DINOv2 VPR 특징 기반 프레임 간 유사도(추가 추론 비용 제로), 매칭 실패 시 이전 프레임 성공 결과를 지능적으로 재사용
- **🔭 Lookahead 이중 확인**: 단계 전환 시 VPR 위치 확인과 다음 단계 서브이미지 매칭을 동시 검증하여 조기 advance 방지
- **📤 통합 출력 형식**：기억 모드 온/오프에 관계없이 일관된 응답 형식, 항상 `pixel_target` 제공
- **🚧 YOLOv8n 차폐 감지**：서브 이미지 매칭 실패 시 카메라 화면의 차폐를 자동 감지(VPR 결과와 독립), 차폐 시 정지 대기 후 해소되면 내비게이션 재개
- **🚦 횡단보도 신호등 감지** (**v2.6.0**)：from/to 노드 이름이 모두 "路口"를 포함하는 단계 구간에서 camera_1 / camera_2 에 대해 YOLO11n + 상하 2단 HSV 색상 판별 + 시계열 평활화로 횡단보도 신호등을 감지. 빨간불 시 로봇이 제자리 대기(최우선, 서브 이미지 / Qwen 결정보다 먼저 return), 초록/없음이면 일반 내비게이션 수행. 구간을 벗어나면 상태 자동 초기화
- **🤖 Qwen3.5 폴백 그라운딩**：2단계 추론(존재 확인 + 조건부 그라운딩)으로 Qwen3.5-9B VLM 폴백
- **📷 어안 왜곡 보정**：시작 시 `cam/params.yaml`에서 카메라 내부 파라미터 로드, 원통형 투영 왜곡 보정 적용
- **🧭 픽셀→로봇 좌표 변환**：정규화된 `pixel_target`을 완전 물리 파이프라인으로 로봇 운동 좌표로 변환
- **🔄 측면 카메라 회전 처리**：camera_3/camera_4 매칭 성공 시 목표 방향으로의 회전 동작 자동 출력
- **🌐 WebSocket 서비스**：실시간 스트리밍으로 이미지 수신 및 내비게이션 명령 반환
- **⚙️ 통합 설정 관리**：모든 VPR 파라미터를 `deploy/vpr_config.yaml`에 집중 관리

---

## 🏗️ 시스템 아키텍처

```
memory-nav/
├── memory_nav/                     # 핵심 메모리 내비게이션 모듈
│   ├── memory_navigator.py         # 내비게이터 메인 인터페이스
│   ├── memory_models.py            # 데이터 모델 (Node, Edge, Plan, VPRResult)
│   ├── memory_graph.py             # 토폴로지 그래프 (BFS/Dijkstra)
│   ├── memory_vpr.py               # VPR 매칭 엔진 (순환 시프트 + 순서 불변)
│   ├── memory_builder.py           # 메모리 빌더
│   ├── sub_image_matcher.py        # 서브 이미지 매처 (DINOv3 밀집 특징, online_mapper 도 재사용)
│   ├── occlusion_detector.py       # YOLOv8n 차폐 검출기
│   ├── traffic_light_detector.py   # YOLO11n 신호등 검출 (횡단보도 구간 활성화)
│   ├── fisheye_undistort.py        # 어안 왜곡 보정 (원통형 투영)
│   ├── coord_transform.py          # 픽셀→로봇 좌표 변환
│   ├── qwen35_point_grounder.py    # Qwen3.5 그라운딩 (폴백)
│   ├── qwen35_grounding_server.py  # Qwen3.5 서브프로세스 추론 서버
│   ├── vpr_factory.py              # VPR 추출기 팩토리
│   ├── vpr_config_loader.py        # 통합 설정 로더
│   └── selavpr_model/              # SelaVPR++ 모델 코드
├── deploy/                         # 배포 엔트리
│   ├── ws_proxy_with_memory.py     # WebSocket 프록시 서비스 (메인, 의도 라우팅 포함)
│   ├── vpr_config.yaml             # VPR 통합 설정 파일 (selavpr 임계값 0.56)
│   ├── build_memory.sh             # 메모리 구축 스크립트
│   ├── start_qwen_vllm.sh          # Qwen3.5-9B vLLM 런처 (GPU 1, 포트 8199)
│   ├── start_qwen08_vllm.sh        # Qwen3.5-0.8B vLLM 런처 (GPU 0, 포트 8198)
│   └── start_server.sh             # 서버 시작 스크립트
├── cam/                            # 다안 어안 카메라
│   ├── params.yaml                 # 카메라 파라미터
│   └── tools/                      # 독립 도구
├── scripts/
│   └── memory_visualization_server.py  # 시각화 서비스 (서브 이미지 + 포인팅 + 차폐 감지)
├── pretrained/                     # 사전 학습 모델 (YOLOv8n, DINOv3 등)
├── merged_labeled_data/            # 메모리 어노테이션 데이터
├── online_mapper/                  # 🛰️ 온라인 능동 맵 생성 모듈 (3 계층)
│   ├── run_online_map.py           # CLI 엔트리
│   ├── config.py                   # 글로벌 설정 (depth/vo/occ_backend 스위치)
│   ├── core/online_mapper_core.py  # ⭐ 메인 오케스트레이터 (~870 줄)
│   ├── geometry/                   # Geometry 계층 (VGGT-1B 기하 프론트엔드)
│   │   ├── vggt_backend.py         # ⭐ VGGT-1B 싱글톤 + 슬라이딩 윈도우 (NEW v2.2)
│   │   ├── depth_estimator.py      #   DA-V2 + VGGTDepthEstimator + 팩토리
│   │   ├── visual_odometry.py      #   MonoVO + VGGTVisualOdometry + 팩토리
│   │   ├── pose_graph.py           #   scipy LM 포즈 그래프
│   │   ├── junction_detector.py    #   4 카메라 깊이 교차로 (stateless)
│   │   ├── traversability.py       # ⭐ VGGT 점군 → 픽셀 단위 통행가능도 (resolve_crop_point)
│   │   └── occupancy.py            #   1D ray-cast + dense 점군 직접 채움
│   ├── topology/                   # Topology 계층
│   │   ├── keyframe_selector.py    #   다중 트리거 키프레임 선택
│   │   ├── loop_closure.py         #   자동 임계값 + ORB 기하 검증
│   │   ├── connection_builder.py   #   ⭐ next_positions: 기하 사전 + traversability + person 페널티
│   │   ├── auto_sub_image_extractor.py  # 그라운딩 crop (memory_nav DINOv3Strategy 재사용)
│   │   └── graph.py                #   TopoGraph / TopoNode
│   ├── semantics/                  # Semantics 계층
│   │   ├── open_set_detector.py    #   Grounding-DINO 래퍼
│   │   ├── door_plate_tracker.py   #   도어플레이트 대표 프레임 선택
│   │   ├── hallucination_filter.py # ⭐ STRICT 프롬프트 + QwenVerifier + MultiFrameVoter
│   │   ├── node_category.py        # ⭐ 노드 카테고리 분류기 + CN/EN 매핑
│   │   ├── node_naming.py          # ⭐ 구조화 명명 NodeName (NEW v2.3)
│   │   ├── colocation_merger.py    # ⭐ 동일 위치 병합 (NodeName.merge_names 사용)
│   │   ├── auto_landmark_namer.py  # Qwen3.5 장면 명명 (vLLM)
│   │   └── scene_graph.py          #   계층적 씬 그래프
│   ├── vpr/                        # VPR 계층
│   │   └── node_distance_estimator.py   # VPR 노드 거리 추정
│   ├── viz/                        # 시각화 계층
│   │   └── visualize.py            #   finalize 시 pose_graph.png / occupancy.png / keyframe_timeline.png / scene_overview.txt 생성
│   └── io/
│       └── merged_data_writer.py   #   출력 라이터 + 구조화 필드
├── third_party/vggt_space/         # VGGT 소스 (.gitignore, HF Space에서 다운로드)
├── pretrained/                     # 모델 가중치 (.gitignore)
│   ├── vggt-1b/                    #   facebook/VGGT-1B
│   ├── depth-anything-v2-small-hf/ #   백업 depth backend
│   ├── grounding-dino-base/        #   IDEA-Research/grounding-dino-base
│   ├── dinov3_vitb16.safetensors   #   VPR 백본
│   ├── yolov8n.pt                  #   차폐 감지
│   └── yolo11n.pt                  #   횡단보도 신호등 감지
├── tests/
│   └── test_memory_ws.py           # WebSocket 통합 테스트
└── docs/
    └── online_mapper.md            # 📘 online_mapper 전체 설계 문서
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
  ├─ 신호등 감지 ("路口→路口" 스텝에서만 활성화, camera_1+camera_2):
  │   └─ 빨간불 → action=[0,0,0] 제자리 대기 (최우선, 즉시 return)
  │      초록/none → 후속 판단으로
  │      비횡단보도 스텝 → reset_state, 건너뜀
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

## 🎯 4 클래스 의도 라우팅

기억 내비게이션에 진입하기 전, 서버는 Qwen3.5-0.8B vLLM 으로 모든 요청을 분류하고 `task` 를 네 경로 중 하나로 라우팅합니다.

| 의도 | 예시 트리거 | 핸들러 | 응답 형태 |
|------|-------------|--------|-----------|
| `navigate` | "C8 前台 去一下", "D 동으로 데려다줘", "출발 지점 복귀" | 기억 내비게이션 메인 플로우 | `action=[x,y,yaw]` + `memory_info` |
| `ask_location` | "내가 어디에 있나", "현재 위치", "지금 위치는" | `handle_ask_location` | `action=[0,0,0]` + `response_text` |
| `ask_direction` | "D 동으로 어떻게 가나", "로비 가는 길" | `handle_ask_direction` | `action=[0,0,0]` + `response_text` |
| `mapping` | `"mapping"` / `"开始建图"` / `"启动扫图"` / `"停止建图"` / `"完成扫图"` | 매핑 세션 생명주기 (시작/종료는 키워드로 판정) | `mode="mapping"` + `log` / `mapping` 또는 `summary` |

백엔드 우선순위: **Qwen3.5-0.8B (포트 8198)** → Qwen3.5-9B (포트 8199) 폴백 → 키워드 룰 폴백. 분류당 약 50 ms. 17/17 분류 정확도 측정 (매핑 자연어 6 개 포함).

### ask_location 응답 예시

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

VPR 유사도가 임계값 미만일 경우, 핸들러는 유사도 top-2 노드를 "A 와 B 사이"로 응답합니다:

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

### ask_direction 응답 예시

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

경로 안내는 Qwen3.5-0.8B 가 다듬어 작성하며, LLM 실패 시 핸들러가 문자열 템플릿으로 폴백합니다.

### 중단 하 내비게이션 연속성

ask_location 과 ask_direction 핸들러는 `nav_state.plan` / `current_step_idx` / `last_task` 를 **변경하지 않으므로**:

| 프레임 | task | 동작 |
|--------|------|------|
| 0 | `"前往C8前台"` | 내비게이션 시작, plan 구축 |
| 1..N-1 | `null` | `last_task` 재사용, 진행 계속 |
| K | `"现在在什么位置"` | 현재 위치 응답, `nav_state` 유지 |
| K+1 | `null` | 보존된 스텝부터 계속, phase=verifying |
| M | `"去 X 怎么走"` | 경로 응답, `nav_state` 유지 |
| M+1 | `null` | 원래 plan 계속 수행 |

응답의 `nav_preserved` 블록으로 UI 에서 원래 내비게이션 태스크가 아직 활성 상태임을 확인할 수 있습니다.

---

## 🛰️ 온라인 맵 생성 (online_mapper)

`online_mapper/` 는 memory-nav 의 스트리밍 온라인 능동 맵 생성 모듈입니다. VGGT-1B 단일 추론으로 depth / pose / dense 포인트 클라우드를 동시에 얻고, `OnlineMapperCore` 가 기하 / 토폴로지 / 시맨틱 세 계층을 조율해 고품질 시맨틱 토폴로지 그래프를 생성합니다.

- **⚙️ 기하 프런트엔드**: VGGT-1B 슬라이딩 윈도우 4 프레임 bf16 싱글톤; `VisualOdometry` 는 VGGT extrinsics 를 재사용하여 추가 추론 없음; `OccupancyGrid` 는 dense 포인트 클라우드로 직접 채움; `Traversability` 는 포인트 클라우드에서 지면 평면 주행 가능도를 추정하고 `resolve_crop_point` 로 crop 중심을 walkable segment 중앙으로 밀어냄 (`detect_vertical_obstacle_columns` 으로 기둥 열 마스크 적용, 화면 하단 절반만 스캔하여 천장 오판 방지)
- **🕸️ 토폴로지**: 다중 트리거 키프레임 (VPR + 병진 + 회전 + 정보 이득 + 교차점 + 시맨틱 화이트리스트); 매 프레임 전역 VPR + ORB 기하 검증 루프 클로저; `ConnectionBuilder` 가 `next_positions` 에 기하 방향 사전 (동일 segment ALPHA=0.5 / 300 s 진짜 단절 ALPHA=0, motion-heading 을 `pose.theta` 보다 우선, 역방향 하드 페널티) + traversability 통로 보정 + GroundingDINO 인물 차폐 페널티 + `cx` 가장자리 하드 제약을 결합; finalize 시 spatial / temporal KNN 로 이웃을 재구축하며 cross-gap filter (> 60s 간격이고 bridging keyframe 없으면 시간 엣지 거부) 포함; DINOv3 서브 이미지 매칭은 `memory_nav.sub_image_matcher.DINOv3Strategy` 를 직접 재사용
- **🧠 시맨틱**: 멀티 카메라 `describe_scene` 투표 (≥2 cam 일치 시 노드 생성, 4 cam 전원 불일치면 skip) + temporal consensus (FUNCTION_AREA canonical / CROSS / T_JUNCTION 은 2/4 에서 단일 프레임 면제, 그 외에는 최근 3 프레임 `_recent_scene_winners` 에 나타나야 verified, LANDMARK_FACILITY 는 bypass 안함); 문패는 STRICT prompt + Qwen 2 차 검증 + `MultiFrameVoter`, `BUILDING_LANDMARK` 는 votes ≥ 4 필수 (글자 OCR 환각 차단) 이며 숫자 환각 단일 프레임 fast-pass 비활성; canonical 정규화 (电梯口 / 电梯间 → 电梯厅; 快递柜 / 外卖柜 / 储物柜 / 智能取餐柜 → 外卖柜区); `_merge_by_canonical_name` 으로 동일 base 강제 머지 + 3 m 공간 클러스터 가드 + latest-anchor; `NodeName` 가 구조화 `category · organization` 이름 생성; `ColocationMerger` 는 카테고리 불일치 가드 + 이른 anchor tie-break; 문패 2 단계 귀속 (functional 을 먼저 생성, brand 는 나중에 attach, `RELOCATE-DISPLAY` 규칙이 타깃 노드의 출처에 따라 display 프레임 재배치 여부 결정)
- **🎯 출력**: `merged_labeled_data/<id>/node_position_info.json` (구조화 `self_position` + `next_positions` + crops), `scene_graph.json`, `pose_graph.json`, `metrics.json`, `online_mapping_log.jsonl`, `plate_voter_dump.json`
- **🔁 레퍼런스 실행**: `memory_test_data` 캠퍼스 주행 281 프레임 → 8 노드 체인 `电梯厅 → 前台 → C座 → H座电梯 → A座 → B座入口 → 外卖柜区·EXHIOH → 2号外卖柜`

### WebSocket 듀얼 모드

`deploy/ws_proxy_with_memory.py` 는 **9528 포트**에서 수신하며, 단일 연결로 **두 가지 모드**를 지원합니다. 모든 요청은 동일한 형태 `{id, task, pts, images}` 를 유지하고, 라우팅은 **4 클래스 의도 라우팅** (`navigate / ask_location / ask_direction / mapping`) 으로 구동됩니다.

| `task` 값 | 의도 | 효과 |
|-----------|------|------|
| `"mapping"` / `"开始建图"` / `"启动扫图"` … | `mapping` (시작) | 맵핑 모드 진입 / 유지. 첫 프레임에서 `MappingSession` 자동 생성, 이후 프레임이 `OnlineMapperCore` 에 입력 |
| `"stop_mapping"` / `"停止建图"` / `"完成扫图"` … | `mapping` (종료) | `finalize` + 시각화 트리거, 요약 반환 후 nav 모드로 복귀 |
| 내비게이션 / ask-location / ask-direction | `navigate` / `ask_location` / `ask_direction` | 기억 내비게이션 실행. 세션이 맵핑 중이었다면 먼저 자동 finalize 후 nav 모드로 전환 |

제어 명령(`{"command": "..."}`)은 상태 조회 전용으로만 유지됩니다: `mapping_status` / `memory_status` / `session_status` / `reset` / `reset_memory` / `toggle_memory`.

맵핑 모드에서는 각 `{id, task:"mapping", pts, images: {camera_1..4}}` 프레임이 `process_mapping_frame` → `OnlineMapperCore.process_frame` 로 라우팅되고 `finalize` 시 flush 됩니다. SelaVPR 추출기는 `MemoryNavigator.extractor` 를 통해 nav / mapping 두 모드 간에 **공유**되어 이중 로드를 방지합니다.

- **산출물 경로**: `deploy/logs/mapping_output/session_{ts}_{client_id}/` (`online_mapper/output/` 와는 별개이며, 후자는 `run_online_map.py` 베이스라인용)
- **임시 프레임 디렉토리**: `deploy/logs/mapping_frames/session_*/`, finalize 시 자동 정리
- **연결 끊김 시 자동 finalize** 로 데이터 보존

---

## 🚀 빠른 시작

```bash
git clone https://github.com/jx1100370217/memory-nav.git
cd memory-nav
pip install -r requirements/core_requirements.txt
pip install -e .

# 기억 구축
bash deploy/build_memory.sh

# 서비스 시작
# 1. Qwen3.5-9B vLLM 기동 (폴백 그라운딩 + 맵핑 명명)
bash deploy/start_qwen_vllm.sh 1 8199

# 2. Qwen3.5-0.8B vLLM 기동 (의도 분류 + 경로 안내)
bash deploy/start_qwen08_vllm.sh 0 8198

# 3. 메인 서비스 기동 (deploy/vpr_config.yaml 자동 로드)
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

## 🧪 테스트

통합 테스트 스크립트 `tests/test_memory_ws.py` 는 두 모드를 모두 지원합니다:

```bash
python -m pytest tests/unit_test/test_basic.py -v

# 내비게이션 재생 (기본값): 첫 프레임은 TASK 전체를 전송하고, 이후 프레임은
# task=None 을 전송 (서버가 last_task 재사용). 시퀀스 상에 세 번의 ask_*
# 인터럽트를 균등 주입하여 원래 nav_state 가 보존되는지 검증한다.
python tests/test_memory_ws.py
python tests/test_memory_ws.py --mode nav

# 맵핑 재생 — 자동으로 task="mapping" → 전체 프레임 입력 → task="stop_mapping",
# 토폴로지 / 키프레임 / 도어플레이트 / 런타임 분해 + 산출물 경로 출력
python tests/test_memory_ws.py --mode mapping
```

---

## 📋 업데이트 로그

### v2.6.0

- **🚦 횡단보도 신호등 감지**: 신규 `memory_nav/traffic_light_detector.py` — YOLO11n 검출 + 상하 2단 HSV 색상 판별(상단 빨강 / 하단 초록) + 시계열 상태 머신
  - `from_node` 와 `to_node` 이름이 모두 "路口" 를 포함하는 단계에서만 활성화 (camera_1 + camera_2 전방)
  - 빨간불 시 즉시 `action=[0,0,0]` (제자리 대기) 반환, 서브 이미지 / Qwen 결정보다 최우선
  - 구간을 벗어나면 `reset_state` 자동 호출하여 색상 잔류 방지
  - 각도 필터(|global_angle| ≤ 30°) + bbox 높이 필터(≤ 120 px) 로 측후방/근거리 신호등 오감지 차단
  - 응답에 최상위 `traffic_light` 필드 추가; `memory_info.phase = "red_light"` 으로 정지 프레임 표시
- **🛰️ online_mapper crop 파이프라인 리팩터**:
  - 신규 `online_mapper/geometry/traversability.py:resolve_crop_point` 가 Qwen crop 중심을 **유지하지 않고 walkable segment 중앙으로 밀어냄**, obstacle 가장자리에서 단일 점은 walkable 이지만 crop 반경 안에 obstacle 이 있는 케이스(녹지벽 / 개찰구 기둥)를 수정
  - 신규 `detect_vertical_obstacle_columns(bottom_frac=0.5)` 기둥 열 검출은 **화면 하단 절반만 스캔**하여 2.5 m 천장이 모든 열을 obstacle 로 오판하는 문제 회피
  - 기하 투영 fallback 도 traversability 검증 통과 (target cx 에 기둥이 투영되면 Qwen 원점으로 복귀)
  - ConnectionBuilder 기하 사전 ALPHA 2 단 분할: 동일 segment `0.5` (구버전 0.2, 기하 사전 주도), 300 s 진짜 단절 `0` (geo_bonus 0 으로 클리어, 순수 visual sim 으로 판단하여 VO 드리프트 오염 방지)
  - DINOv3 서브 이미지 매칭은 `memory_nav.sub_image_matcher.DINOv3Strategy` 를 직접 재사용, 코드 중복 제거
- **🛰️ online_mapper 노드 머지 개선**:
  - `_merge_by_canonical_name` 에 3 m 공간 클러스터 가드 추가: 동일 canonical 이름이라도 유클리드 거리 > 3 m 이면 머지 안함 (시작 엘리베이터홀 vs H 동 옆 엘리베이터가 모두 "电梯厅" 으로 정규화되어 잘못 머지되던 문제 수정)
  - anchor 를 latest `frame_idx` 로 변경 (경로상 landmark 통과 후 도달한 keyframe 이 일반적으로 landmark 에 더 가깝고 plate bbox 도 더 큼)
- **🛰️ online_mapper temporal consensus 강화**:
  - LANDMARK_FACILITY 단일 프레임 화이트리스트 fast-pass 제거, N11 电梯厅\_2 형 멀티 카메라 공동 환각 차단
  - CROSS / T_JUNCTION 장면은 2/4 consensus 로 temporal 면제 (前台/关爱室류 landmark 의 단일 프레임 인식 구제)
  - FUNCTION_AREA canonical winner 는 2/4 consensus 로 temporal 면제 (FUNCTION_AREA 는 Qwen 저환각 구체 landmark)
- **🛰️ online_mapper plate 투표**:
  - BUILDING_LANDMARK plate 는 votes ≥ 4 일 때 confirm, Qwen OCR 글자 환각 차단 (`13号楼` / `D座` 등 환각은 보통 votes ≤ 3, 최소 실제 `B座` = 4)
  - BUILDING_LANDMARK 단일 프레임 화이트리스트 fast-pass 비활성화

### v2.5.1

- **🗺️ 온라인 맵핑을 4 번째 의도 클래스로 승격**: 의도 분류를 3 → 4 클래스로 확장 (navigate / ask_location / ask_direction / **mapping**)
  - "开始建图" / "启动扫图" / "请开始建图" 같은 자연어 발화가 자동으로 `MappingSession` 생성
  - "停止建图" / "结束建图" / "完成扫图" 같은 자연어 발화가 자동으로 `finalize` 트리거
  - 하드코딩된 `task="mapping"` / `task="stop_mapping"` 은 하위 호환 (LLM 호출 0)
  - 4 클래스 의도 분류 테스트 셋에서 17/17 분류 정확도 측정
- **🔧 `handle_client` 통합 디스패치**: 의도 분류를 `process_inference_with_memory` 에서 `handle_client` 최상단으로 올려 intent 값으로 직접 라우팅; `process_inference_with_memory` 에 `intent` 매개변수를 추가하여 중복 분류 방지

### v2.5.0

- **🎯 의도 분류 라우팅**：Qwen3.5-0.8B vLLM 기반 신규 `IntentClassifier`. 모든 `task` 를 navigate / ask_location / ask_direction 로 자동 분류, 호출당 약 50 ms
  - 백엔드 우선순위: Qwen3.5-0.8B (8198) → Qwen3.5-9B (8199) 폴백 → 키워드 룰 폴백
  - 신규 런처 `deploy/start_qwen08_vllm.sh` (GPU 0, 약 4.6 GB 메모리 사용)
- **📍 현재 위치 질의 (`handle_ask_location`)**："현재 위치"·"내가 어디에 있나" 질의에 VPR 만 실행하여 `response_text="当前的位置是 X"` 반환. VPR 유사도 저하 시 유사도 top-2 노드를 "A 와 B 사이"로 응답
- **🧭 길 안내 질의 (`handle_ask_direction`)**：VPR 출발지 + `find_destination` 목적지 + `plan_navigation` 경로를 조합하고, Qwen3.5-0.8B 가 자연스러운 중국어 문장으로 경로를 서술 (템플릿 폴백 포함)
- **🔁 내비게이션 연속성**：ask_location / ask_direction 은 `nav_state` / `session_state['last_task']` 를 절대 변경하지 않음. 다음 프레임에서 `task=None` 만 보내면 원래 plan 을 이음매 없이 재개
- **🎛️ 임계값 튜닝**：VPR `similarity_threshold.selavpr` 0.60 → **0.56**; `VPR_ARRIVE_THRESHOLD` 0.70 → **0.68**. 테스트 셋에서 48/49 프레임 VPR 히트 + 내비게이션 완주 첫 달성
- **🧪 test_memory_ws.py 강화**：첫 프레임은 TASK 전체 전송, 이후 프레임은 `task=None` 전송 (`last_task` 재사용). 세 번의 ask_* 인터럽트를 균등 주입하여 `nav_preserved` 100 % 검증
- **🐛 버그 수정**：`process_inference_with_memory` 내부에 남아 있던 모듈 레벨 import 를 가리는 로컬 `import math` 제거

### v2.3.0

- **🛰️ 온라인 능동 맵 생성 모듈** (`online_mapper/`): 3 계층 아키텍처 (Geometry + Topology + Semantics) 기반의 스트리밍 온라인 맵 생성 모듈 신설
  - **Geometry 계층**: 단안 ORB+EssentialMatrix VO, Depth-Anything-V2, scipy LM 포즈 그래프, 2D 점유 그리드, 4-카메라 깊이 교차로 감지
  - **Topology 계층**: 다중 트리거 키프레임, 자동 임계값 + ORB 기하 검증 전역 루프 클로저, 공간 KNN ∪ 시간 인접 인접성 재구축
  - **Semantics 계층**: STRICT 프롬프트 + QwenVerifier 2차 검증 + MultiFrameVoter 다중 프레임 투표 + 부분 문자열 변이 병합 + 7 카테고리 화이트리스트 + ColocationMerger 동일 위치 병합 + CN/EN 이중 언어 명명 + NameDeduplicator 접미사 중복 해결
  - `merged_labeled_data/` 스키마 생성, 추가로 `scene_graph.json` / `pose_graph.json` / `online_mapping_log.jsonl` / `metrics.json` 생성
  - **테스트 데이터 (49 프레임) 최종 결과**: 고품질 노드 5 개 (인쇄 구역 / 리셉션 / NEUMANN 전기실 / 케어 룸 / DEEPROUTE.AI 리셉션), 환각 0 / 중복 0 / 루프 클로저 2 회, validator 5/5 통과
  - 전체 설계 문서: **[`docs/online_mapper.md`](docs/online_mapper.md)**
  - 이터레이션 기록 (r1→r6): [`online_mapper/RESULTS.md`](online_mapper/RESULTS.md)
- **🌐 WebSocket 듀얼 모드**: `deploy/ws_proxy_with_memory.py` 가 nav 와 mapping 두 모드를 동시 지원. 클라이언트가 `start_mapping` / `stop_mapping` / `mapping_status` 로 전환. SelaVPR 모델 공유, 산출물은 `deploy/logs/mapping_output/session_{ts}_{cid}/`, 연결 끊김 시 자동 finalize.
- **🧪 병합된 테스트 스크립트**: `tests/test_memory_ws.py --mode {nav,mapping}` 로 통합. 구 `deploy/test_mapping_client.py` 삭제됨.
- **📊 online_mapper 시각화**: `online_mapper/viz/visualize.py` 추가, finalize 시 `pose_graph.png` / `occupancy.png` / `keyframe_timeline.png` / `scene_overview.txt` 생성.
- **🔧 `OnlineMapperCore` API 분리**: `run()` → 공개 `process_frame` + `finalize` 로 스트리밍 지원.

### v2.2.0

- **🆕 자동 맵 생성 모듈**：이미지 시퀀스에서 위상 그래프를 자동 생성하는 모듈 신설 (v2.3.0 에서 `online_mapper/` 로 통합)
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
