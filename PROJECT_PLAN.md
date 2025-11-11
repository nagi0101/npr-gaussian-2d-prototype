# 3DGS 기반 회화적 브러시 2D 프로토타입 프로젝트 계획서

## 1. 프로젝트 개요

### 1.1 목표

Painting with 3D Gaussian Splat Brushes의 stamp-based painting 방법론을 활용한 회화적 브러시 시스템의 최소 기능 프로토타입(MVP) 구현

### 1.2 핵심 특징

-   3D Gaussian Splatting을 z=0 평면에 제한하여 2D 회화 효과 구현
-   브러시 스트로크를 여러 가우시안 블롭의 집합으로 회화적 브러시 질감 표현
-   단일 색상(회색) 렌더링으로 시작하여 기본 기능 검증

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
├── Backend (Python/FastAPI)
│   ├── Gaussian Manager
│   ├── Brush Engine
│   ├── Rendering Engine
│   └── WebSocket Handler
├── Frontend (HTML/JS)
│   ├── Canvas Interface
│   ├── Mouse Event Handler
│   └── WebSocket Client
└── Shared
    └── Gaussian Representation
```

### 2.2 핵심 컴포넌트

#### 2.2.1 Gaussian Representation

```python
class Gaussian2D:
    def __init__(self):
        self.position = np.array([x, y, 0])  # z=0 고정
        self.scale = np.array([sx, sy, sz_min])  # z 스케일 최소화
        self.rotation = np.array([0, 0, 0, 1])  # quaternion
        self.opacity = 0.8
        self.color = np.array([0.5, 0.5, 0.5])  # 회색
```

#### 2.2.2 Brush Definition

```python
class BrushStamp:
    def __init__(self):
        self.gaussians = []  # 상대 위치의 가우시안 집합
        self.size = 1.0
        self.spacing = 0.2
        self.texture_pattern = None  # 향후 확장용
```

## 3. 구현 단계

### Phase 1: 기본 인프라 구축 (2일)

#### 1.1 Backend 설정

-   FastAPI 서버 구축
-   WebSocket 통신 설정
-   기본 Gaussian 데이터 구조 구현

#### 1.2 Frontend 설정

-   HTML5 Canvas 기반 인터페이스
-   마우스 이벤트 처리
-   WebSocket 클라이언트

#### 1.3 2D Gaussian Renderer

```python
def render_gaussian_2d(gaussian, viewport):
    # 3DGS의 2D 투영 공식 적용
    # z=0 평면에서 단순화된 계산
    cov2d = compute_2d_covariance(gaussian)
    color = gaussian.color * gaussian.opacity
    return rasterize_ellipse(cov2d, color)
```

### Phase 2: 브러시 시스템 구현 (3일)

#### 2.1 Brush Stamp Generation

```python
def create_brush_stamp(pattern_type='circle', num_gaussians=10):
    stamp = BrushStamp()
    # 원형, 사각형 등 기본 패턴 생성
    positions = generate_pattern(pattern_type, num_gaussians)
    for pos in positions:
        g = Gaussian2D()
        g.position = pos
        stamp.gaussians.append(g)
    return stamp
```

#### 2.2 Stamp Placement (Rigid Transform)

```python
def place_stamp(stamp, mouse_pos, stroke_tangent):
    # 1. 마우스 위치로 이동
    # 2. 스트로크 접선 방향 정렬
    transformed_stamp = []
    for g in stamp.gaussians:
        g_new = g.copy()
        g_new.position = rotate_and_translate(g.position, mouse_pos, stroke_tangent)
        transformed_stamp.append(g_new)
    return transformed_stamp
```

#### 2.3 Stroke Spline Fitting

```python
def fit_stroke_spline(mouse_points):
    # 마우스 포인트들을 cubic spline으로 보간
    spline = CubicSpline(mouse_points)
    # stamp 배치 위치 계산
    stamp_positions = []
    for t in np.arange(0, 1, stamp_spacing):
        pos = spline(t)
        tangent = spline.derivative(t)
        stamp_positions.append((pos, tangent))
    return stamp_positions
```

### Phase 3: 고급 변형 구현 (3일)

#### 3.1 Non-rigid Deformation

```python
def deform_stamp_2d(stamp, curvature):
    # 2D에서 단순화된 변형
    # Painting 논문의 spline 곡률 기반 변형을 2D로 적용
    deformed = []
    for g in stamp:
        g_new = g.copy()
        # 곡률에 따른 위치 조정
        offset = compute_curvature_offset(g.position, curvature)
        g_new.position += offset
        # 타원 모양 조정
        g_new.scale *= (1 + curvature_factor)
        deformed.append(g_new)
    return deformed
```

#### 3.2 Simplified Inpainting

```python
def inpaint_overlap_2d(stamp1, stamp2):
    # 2D에서 단일 뷰만 고려
    overlap_region = find_overlap(stamp1, stamp2)
    if overlap_region:
        # 간단한 블렌딩 또는 opacity 조정
        for g in overlap_region:
            g.opacity *= 0.7  # 겹치는 부분 투명도 조정
    return stamp1 + stamp2
```

### Phase 4: 통합 및 최적화 (2일)

#### 4.1 실시간 렌더링 파이프라인

```python
class RealTimeRenderer:
    def __init__(self):
        self.gaussian_buffer = []
        self.render_queue = Queue()

    def add_stroke(self, stroke_gaussians):
        self.gaussian_buffer.extend(stroke_gaussians)
        self.trigger_render()

    def render_frame(self):
        # GPU 가속 활용 (RTX 3090)
        image = gaussian_splatting_2d(self.gaussian_buffer)
        return image
```

#### 4.2 WebUI 통합

```javascript
// Frontend canvas handling
class BrushCanvas {
    constructor() {
        this.canvas = document.getElementById("canvas");
        this.ctx = this.canvas.getContext("2d");
        this.ws = new WebSocket("ws://server:8000/paint");
        this.isDrawing = false;
        this.strokePoints = [];
    }

    onMouseDown(e) {
        this.isDrawing = true;
        this.strokePoints = [{ x: e.clientX, y: e.clientY }];
    }

    onMouseMove(e) {
        if (this.isDrawing) {
            this.strokePoints.push({ x: e.clientX, y: e.clientY });
            this.ws.send(
                JSON.stringify({
                    type: "stroke_update",
                    points: this.strokePoints,
                })
            );
        }
    }
}
```

## 4. 기술 스택

### Backend

-   Python 3.9+
-   PyTorch (CUDA 12.2 호환)
-   FastAPI + WebSocket
-   NumPy/SciPy (spline fitting)
-   diff-gaussian-rasterization (수정 버전)

### Frontend

-   HTML5 Canvas
-   Vanilla JavaScript
-   WebSocket API

## 5. 개발 환경 설정

### 5.1 Docker 컨테이너 설정

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
RUN pip install fastapi uvicorn websockets numpy scipy
RUN pip install diff-gaussian-rasterization
EXPOSE 8000
```

### 5.2 프로젝트 구조

```
brush-gaussian-2d/
├── backend/
│   ├── core/
│   │   ├── gaussian.py
│   │   ├── brush.py
│   │   └── renderer.py
│   ├── api/
│   │   └── websocket.py
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── canvas.js
│   └── style.css
├── requirements.txt
└── Dockerfile
```

## 6. 주요 구현 과제 및 해결 방안

### 6.1 실시간 성능

-   **문제**: 많은 수의 가우시안 실시간 렌더링
-   **해결**: RTX 3090 GPU 활용, 배치 처리, LOD 시스템

### 6.2 WebSocket 지연

-   **문제**: 네트워크 지연으로 인한 반응성 저하
-   **해결**: 클라이언트 측 예측, 델타 업데이트 전송

### 6.3 메모리 관리

-   **문제**: 계속 추가되는 가우시안으로 인한 메모리 증가
-   **해결**: 뷰포트 기반 culling, 오래된 스트로크 병합

## 7. 확장 가능성

-   다중 색상 지원
-   텍스처 브러시 패턴
-   3D 확장 (z축 활용)
-   브러시 라이브러리 시스템
-   Diffusion 기반 인페인팅 통합

이 계획서는 빠른 프로토타입 개발을 위한 최소 기능 구현에 초점을 맞추었습니다. 각 단계는 독립적으로 테스트 가능하도록 설계되어 있어, 문제 발생 시 빠른 디버깅이 가능합니다.
