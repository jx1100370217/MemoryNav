<div align="center">

# 🧠 MemoryNav

**시각적 기억 내비게이션 시스템 | Visual Memory Navigation System**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/Version-2.0.0-orange.svg)](https://github.com/jx1100370217/MemoryNav/releases/tag/v2.0.0)

VPR(시각적 장소 인식)과 위상 지도 기반 로봇 기억 내비게이션 시스템

[English](README_EN.md) | [中文](README.md) | [日本語](README_JA.md) | **한국어**

</div>

---

## 📖 소개

MemoryNav는 이동 로봇을 위한 시각적 기억 내비게이션 시스템입니다. 4개의 전방위 어안 카메라로 이미지를 수집하고, VPR 기술을 사용하여 사전 구축된 위상 기억 그래프에서 위치를 추정하며, YOLOv8n 시각적 차폐 감지와 Qwen3.5-9B 시각-언어 모델 폴백을 결합하여 "한번 간 곳을 기억하고 다시 걷는" 기억 내비게이션 능력을 실현합니다.

### 주요 기능

- **🔍 멀티 스킴 VPR 측위**：4가지 SOTA VPR 방법 지원, 설정 파일 하나로 전환 가능
- **🗺️ 위상 기억 그래프**：어노테이션 데이터에서 노드-엣지 위상을 자동 구축, BFS/Dijkstra 최단 경로 계획 지원
- **🔄 순환 시프트 매칭**：4카메라 순환 시프트 알고리즘으로 방향 불변 위치 추정 및 편향각 추정
- **🎯 DINOv3 서브 이미지 매칭**：DINOv3 고밀도 패치 특징의 슬라이딩 윈도우 코사인 유사도 매칭, 실시간 카메라 프레임에서 내비게이션 목표 측위
- **💾 매칭 캐시**：신뢰도가 낮을 경우 마지막 성공 결과를 자동 재사용하여 내비게이션 연속성 향상
- **🔭 Lookahead 이중 확인**: 단계 전환 시 VPR 위치 확인과 다음 단계 서브이미지 매칭을 동시 검증하여 조기 advance 방지
- **📤 통합 출력 형식**：기억 모드 온/오프에 관계없이 일관된 응답 형식, 항상 `pixel_target` 제공
- **🚧 YOLOv8n 차폐 감지**：VPR/서브 이미지 매칭 실패 시 카메라 화면의 차폐(보행자, 물체 등)를 자동 감지, 차폐 시 정지 대기 후 해소되면 내비게이션 재개
- **🤖 Qwen3.5 폴백 그라운딩**：차폐 없이 VPR/서브 이미지 매칭 실패 시 Qwen3.5-9B VLM으로 자동 전환
- **📷 어안 왜곡 보정**：시작 시 `cam/params.yaml`에서 카메라 내부 파라미터 로드, VPR 및 서브 이미지 매칭 전 원통형 투영 왜곡 보정 적용
- **🧭 픽셀→로봇 좌표 변환**：정규화된 `pixel_target`을 원통형 각도·카메라 방위각·우각 파이프라인을 통해 로봇 운동 좌표 `[x_forward, y_lateral, 0.0]`으로 변환
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
│   ├── vpr_factory.py              # VPR 추출기 팩토리
│   ├── vpr_config_loader.py        # 통합 설정 로더
│   └── selavpr_model/              # SelaVPR++ 모델 코드
├── deploy/                         # 배포 엔트리
│   ├── ws_proxy_with_memory.py     # WebSocket 프록시 서비스 (메인 엔트리)
│   ├── vpr_config.yaml             # VPR 통합 설정 파일
│   ├── pretrained/                 # 사전 학습 모델 (YOLOv8n 등)
│   ├── build_memory.sh             # 메모리 구축 스크립트
│   └── start_server.sh             # 서버 시작 스크립트
├── cam/                            # 다안 어안 카메라
│   ├── params.yaml                 # 카메라 파라미터
│   └── tools/                      # 독립 도구
├── scripts/
│   └── memory_visualization_server.py  # 시각화 서비스
├── merged_labeled_data/            # 메모리 어노테이션 데이터
├── tests/
│   └── test_memory_ws.py           # WebSocket 통합 테스트
└── docs/
```

---

## 📷 어안 왜곡 보정

VPR 매칭 및 서브 이미지 매칭 전에 4채널 어안 이미지에 대해 자동으로 원통형 투영 왜곡 보정을 수행합니다.

### 동작 원리

1. 시작 시 `cam/params.yaml`에서 각 카메라의 내부 파라미터(`xi, fx, fy, cx, cy`)와 왜곡 계수(`k1, k2, p1, p2`) 로드
2. 카메라당 한 번 원통형 투영 remap 테이블 사전 계산(`pitch_up` 오프셋 포함)
3. 각 추론 프레임 전 `cv2.remap` 적용, 계산 비용 극히 낮음
4. `cam/params.yaml` 없을 경우 왜곡 보정 스킵, 서비스 정상 동작 유지

```python
from memory_nav.fisheye_undistort import FisheyeUndistorter

undistorter = FisheyeUndistorter.from_yaml("cam/params.yaml")
perspective_images = undistorter.undistort_batch(camera_images)
```

---

## 🧭 픽셀→로봇 좌표 변환

정규화된 `pixel_target: [x_norm, y_norm]`을 전체 물리 파이프라인을 통해 로봇 운동 좌표 `[x_forward, y_lateral, 0.0]`으로 변환합니다.

### 변환 파이프라인

```
x_norm → 원통형 수평각 → + 카메라 방위각 → 글로벌 yaw
y_norm → 원통형 수직각 → 우각 → 거리 추정 (카메라 높이 + pitch_up)
yaw + distance → (x_forward, y_lateral)
```

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

### 동작 원리

1. **기억 구축 시**：각 엣지에 `camera_name`(목표 카메라)과 `crop_image`(어텐션 크롭) 어노테이션
2. **내비게이션 실행 시**：기억에서 crop 서브 이미지를 가져와 실시간 카메라 프레임에 대해 고밀도 특징 매칭 수행
3. **목표 측위**：DINOv3 ViT-B/16으로 고밀도 패치 토큰 추출 → 슬라이딩 윈도우 + unfold 가속 → 코사인 유사도 최대 위치 → `pixel_target`으로 출력
4. **매칭 임계값**：신뢰도 ≥ `SUB_MATCH_CONFIDENCE_THRESHOLD` (현재 0.65) 시 성공
5. **캐시**：낮은 신뢰도 프레임에서 마지막 성공 결과 재사용, 스텝 전환 시 초기화
6. **폴백**：캐시 없을 경우 기억 내 `pixel_box`를 추정값으로 사용

---

## ✨ VPR 방법 비교

| 방법 | 키 | 발표 | 특징 차원 | 백본 | 특징 |
|------|-----|------|---------|------|------|
| **SelaVPR++** ⭐ | `selavpr` | T-PAMI 2025 | 4096D | DINOv2-L + MultiConv | **권장**, 해싱+리랭킹 |
| **MegaLoc** | `megaloc` | CVPR 2025 | 8448D | DINOv2-B + OT | 종합 최강, 다중 데이터셋 SOTA |
| **EffoVPR** | `effovpr` | arXiv 2024 | 3072D | DINOv2-B 다층 CLS | 경량·고속, 실시간 적합 |
| **AnyLoc** | `anyloc` | RA-L 2023 | 가변 | DINOv2-B + VLAD | 클래식, 안정적 |

---

## 🚀 빠른 시작

### 설치

```bash
git clone https://github.com/jx1100370217/MemoryNav.git
cd MemoryNav
pip install -r requirements/base.txt
pip install -e .
```

### 카메라 설정 (선택 사항)

실제 카메라가 있을 경우 캘리브레이션 파일을 `cam/params.yaml`에 배치하세요. 시작 시 어안 왜곡 보정이 자동으로 활성화됩니다.

### 기억 구축

```bash
bash deploy/build_memory.sh
```

### 내비게이션 서비스 시작

```bash
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

## 📋 업데이트 로그

### v2.0.0

- **🆕 YOLOv8n 차폐 감지**：`memory_nav/occlusion_detector.py` 신설
  - YOLOv8n(6MB)으로 person, backpack, umbrella 등 근거리 전경 물체 감지, bbox 면적비 ≥ 35%로 차폐 판정
  - 차폐 시 `action: [0, 0, 0]`(정지 대기), 해소 후 자동 내비게이션 재개
  - 차폐 없을 경우 Qwen3.5 포인팅(landmark_name)으로 폴백 내비게이션
- **🔄 내비게이션 로직 간소화**：기존 트렌드 감지 방식(Case B 건너뛰기 / Case C 재계획 / Case D 유사도 트렌드) 삭제
- **🎯 서브 이미지 매칭 best_fail_camera**：전체 실패 시에도 최고 점수 카메라를 기록, 차폐 감지에 사용
- **🖥️ 차폐 감지 탭**：`memory_visualization_server.py`에 🚧 차폐 감지 검증 탭 추가

### v1.9.0

- **🔭 Lookahead 이중 확인**: 단계 전환 조건을 VPR + 다음 단계 서브이미지 매칭 이중 확인으로 강화
  - 매 프레임마다 현재 단계와 다음 단계를 동시에 서브이미지 매칭
  - VPR이 목표 노드에 매칭되어도 다음 단계 매칭이 성공해야 advance
  - 마지막 단계는 lookahead 없이 직접 advance
- **🎯 서브이미지 매칭 임계값 통합**: `SUB_MATCH_CONFIDENCE_THRESHOLD`를 단일 소스로 전체 파이프라인에 전파
- **📊 테스트 로그 강화**: `la_conf` 열 추가, Lookahead 통계 섹션 추가

### v1.8.0

- **🆕 어안 왜곡 보정**：`memory_nav/fisheye_undistort.py` 추가（`cam/tools/fisheye_undist_cpu.h`에서 포팅）
- **🆕 픽셀→로봇 좌표 변환**：`memory_nav/coord_transform.py` 추가（전체 원통형 투영 파이프라인）
- **🆕 cam/ 디렉토리**：다안 어안 카메라 ROS2 노드 소스코드 및 `params.yaml` 추가

### v1.7.0

- **Qwen3.5 폴백 그라운딩**：InternVLA 대신 Qwen3.5-9B VLM 채택

### v1.6.0

- **서브 이미지 매칭 단순화**：DINOv3 고밀도 특징 매칭만 유지

### v1.5.0

- **통합 출력 형식**：항상 `pixel_target`을 포함하는 일관된 응답 형식

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)를 사용합니다.
