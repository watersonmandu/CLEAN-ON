# CLEAN:ON

1. Python 3.11 또는 3.12 권장. 프로젝트 폴더에서 `python -m pip install -r requirements.txt`
2. `python -m uvicorn server:app --reload`
3. 브라우저 `http://127.0.0.1:8000`
4. 모델 없으면 데모 모드. `dataset/black_spots`, `dataset/clean` 폴더에 각 클래스 이미지 확보 후 `python train.py`.
5. 서버를 재시작하면 `bathroom_model.pth` 로드.

주의: 실험용 모델이며 소량 데이터로 학습한 분류 점수는 임상/위생 판정이 아님. 검증용 데이터는 학습용과 독립적으로 구성하고 중복 이미지를 제거해야 함. 검은 얼룩은 곰팡이 외 다른 오염일 수 있음.
