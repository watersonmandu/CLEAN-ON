# CLEAN:ON (욕실 오염 AI 분석 & 맞춤 청소 가이드)

Render 무료 웹 서비스(512MB RAM) 환경에서 `Out of memory` 없이 안정적으로 구동되도록 경량화(PyTorch -> ONNX Runtime CPU)된 욕실 표면(실리콘, 타일, 줄눈) 오염 패턴 분석 서비스입니다.

---

## 🚀 빠른 시작 (로컬 실행)

### 1. 가상환경 생성 및 의존성 설치
```bash
# 가상환경 생성 및 활성화 (Python 3.11 또는 3.12 권장)
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

# 서버 실행용 경량 의존성 설치 (PyTorch 불필요, 약 50MB 용량)
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python server.py
# 또는
uvicorn server:app --reload --port 8000
```
브라우저에서 `http://127.0.0.1:8000`으로 접속하여 사진 업로드 및 재질 분석을 테스트할 수 있습니다.

---

## 🛠️ ONNX 모델 변환 및 검증 (모델 재학습/변환 시)

### 1. ONNX 변환 스크립트 실행
기존 PyTorch ResNet18 가중치(`bathroom_model.pth`)를 ONNX 포맷(`bathroom_model.onnx`)으로 변환합니다:
```bash
# 모델 변환 시에만 필요한 라이브러리(torch, torchvision, onnx) 설치
pip install -r requirements-train.txt

# 변환 스크립트 실행 (메타데이터 자동 포함)
python convert_to_onnx.py
```
- 생성 파일: `bathroom_model.onnx` (약 42.6MB, GitHub 파일 크기 제한 100MB 이하 충족)
- 동반 메타데이터: `bathroom_model_meta.json` (클래스 순서, 이미지 크기, mean, std)

### 2. PyTorch vs ONNX Runtime 추론 일치성 검증
변환 전 PyTorch 모델과 변환 후 ONNX Runtime CPU 모델의 로짓(Logits) 및 소프트맥스 확률이 오차 1e-6 이내로 일치하는지 자동 검증합니다:
```bash
python test_model_equivalence.py
```

### 3. API 및 웹 기능 통합 테스트
```bash
python verify_api.py
# 또는 HTTP 통합 테스트
python run_test_suite.py
```

---

## 🌐 Render 무료 티어 (512MB RAM) 배포 가이드

Render 무료 인스턴스는 512MB RAM 제한이 있어, 무거운 PyTorch(800MB+ 디스크, 400MB+ RAM)를 불러올 경우 메모리 초과(`Out of memory`)가 발생합니다.  
본 프로젝트는 **ONNX Runtime CPU + NumPy + 단일 워커 설정**을 통해 서버 런타임 메모리를 **약 80~120MB** 수준으로 대폭 절감했습니다.

### 배포 방법 1: Render Blueprint (가장 간단)
1. 프로젝트를 GitHub 저장소에 Push합니다.
2. Render 대시보드에서 **New +** -> **Blueprint** 선택
3. GitHub 저장소를 연결하면 `render.yaml` 설정에 따라 자동으로 서비스가 구성 및 배포됩니다.

### 배포 방법 2: Web Service 수동 생성
1. Render 대시보드 -> **New +** -> **Web Service**
2. GitHub 저장소 선택
3. 기본 설정:
   - **Name**: `clean-on`
   - **Environment**: `Python 3`
   - **Branch**: `main`
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Instance Type**: `Free` (0.1 CPU, 512MB RAM)
4. **Environment Variables (메모리 최적화 환경변수)**:
   - `PYTHON_VERSION`: `3.11.9`
   - `OMP_NUM_THREADS`: `1`
   - `MKL_NUM_THREADS`: `1`
   - `OPENBLAS_NUM_THREADS`: `1`

---

## 📁 주요 프로젝트 파일 구성

- `server.py`: ONNX Runtime CPU 기반 추론 서버 (PyTorch/torchvision 미사용, 메모리 80% 이상 절감)
- `bathroom_model.onnx`: 변환된 경량 ResNet18 ONNX 모델 (~42.6MB)
- `convert_to_onnx.py`: PyTorch 모델(`bathroom_model.pth`)을 ONNX로 변환하는 스크립트
- `test_model_equivalence.py`: PyTorch와 ONNX 모델의 수치 정밀도 및 분류 결과 일치성 검증 스크립트
- `run_test_suite.py` / `verify_api.py`: 엔드포인트 기능 및 유효성 검증 테스트 스위트
- `render.yaml`: Render 무료 서버 자동 배포 명세서 (메모리 절감 옵션 적용)
- `requirements.txt`: 배포 서버용 경량 의존성 (fastapi, uvicorn, onnxruntime, pillow, numpy)
- `requirements-train.txt`: 모델 학습 및 ONNX 변환 전용 의존성
- `index.html`: 반응형 웹 UI (사진 업로드, 재질 선택, 분석 결과, 재질별 청소 가이드)
