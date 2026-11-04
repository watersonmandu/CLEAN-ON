import sys
from pathlib import Path
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms

BASE = Path(__file__).resolve().parent
app = FastAPI(title='CLEAN:ON API')
MODEL_PATH = BASE / 'bathroom_model.pth'

# 모델 및 전처리 초기화
model = None
model_status = 'not_loaded'
model_error_msg = None
classes = ['black_spots', 'clean']
preprocess = None

# 재질별 상세 청소 가이드 정의
CLEANING_GUIDES = {
    'silicone': {
        'title': '실리콘 (Silicone)',
        'description': '실리콘은 탄성이 있고 습기가 오래 머물러 곰팡이가 뿌리를 내리기 쉬운 재질입니다.',
        'steps': [
            '환기팬을 가동하고 창문을 연 뒤, 반드시 방수 고무장갑과 마스크를 착용하세요.',
            '표면의 물기와 먼지를 마른 헝겊으로 완전히 닦아냅니다.',
            '실리콘 전용 젤 타입 세정제 또는 락스를 적신 키친타월을 오염 부위에 밀착해 1~2시간 이상 불려줍니다.',
            '부드러운 스펀지로 살살 닦아내고 미온수로 잔여 세제를 완전히 헹궈냅니다. (거친 철수세미는 실리콘을 손상시킵니다)',
            '마른 수건으로 물기를 닦고 건조 상태를 유지하세요. 실리콘 내부까지 변색이 깊게 침투했다면 실리콘 제거 후 재시공을 권장합니다.'
        ],
        'spots_tip': '실리콘 속으로 파고든 흑색 반점은 표면 세척만으로 완전히 지워지지 않을 수 있습니다. 무리하게 긁어내면 틈새가 벌어져 누수 원인이 되므로, 표백으로 탈색되지 않는다면 재시공을 고려하세요.',
        'clean_tip': '현재 뚜렷한 흑색 오염이 보이지 않는 양호한 상태입니다. 샤워 후 실리콘 주변의 물기를 스퀴지나 마른 수건으로 닦아주면 곰팡이 발생을 예방할 수 있습니다.'
    },
    'tile': {
        'title': '타일 (Tile)',
        'description': '타일은 매끄러운 유약 표면으로 오염 제거가 비교적 쉬우나, 유약 손상에 주의해야 합니다.',
        'steps': [
            '충분한 환기 환경을 확보하고 보호장갑을 착용하세요.',
            '타일 표면에 미온수를 뿌려 가벼운 먼지와 비누 찌꺼기를 1차로 씻어냅니다.',
            '욕실용 중성세제 또는 타일 전용 클리너를 고르게 분사한 후 5~10분간 오염을 불립니다.',
            '부드러운 솔이나 스펀지로 원을 그리듯 문질러 오염을 닦아냅니다.',
            '샤워기로 깨끗하게 헹궈낸 뒤 스퀴지로 벽면/바닥 물기를 쓸어내 건조시킵니다.'
        ],
        'spots_tip': '타일 표면의 검은 점은 비누때와 결합한 곰팡이 포자일 가능성이 높습니다. 강한 연마제나 철수세미를 쓰면 타일 광택이 손상되어 오염이 더 쉽게 달라붙으니 부드러운 도구를 사용하세요.',
        'clean_tip': '타일 표면이 깨끗하게 유지되고 있습니다. 주 1회 중성세제로 가볍게 물청소 후 건조해주시면 오염 축적을 방지할 수 있습니다.'
    },
    'grout': {
        'title': '줄눈 (Grout)',
        'description': '줄눈 시멘트는 미세한 다공성 구조여서 수분과 오염물질을 내부로 쉽게 흡수합니다.',
        'steps': [
            '환풍기를 최대로 켜고 창문을 열어 환기하고, 고무장갑과 눈 보호구를 착용하세요.',
            '줄눈 틈새에 물을 살짝 적신 후 과탄산소다 희석액이나 줄눈 전용 세제를 틈새에 도포합니다.',
            '5~10분 후 낡은 칫솔이나 틈새 전용 브러시를 이용해 줄눈 라인을 따라 부드럽게 문지릅니다.',
            '맑은 물로 세제 거품이 남지 않도록 꼼꼼히 헹굽니다.',
            '수분을 완전히 말려줍니다. 시멘트 줄눈이 파였거나 탈락된 경우 줄눈 보수제나 줄눈 코팅재로 보수하세요.'
        ],
        'spots_tip': '줄눈 내부로 곰팡이가 스며들면 쉽게 지워지지 않습니다. 강산성 세제(염산 계열)는 시멘트를 부식시켜 가루 날림을 유발하므로 절대 피하고, 심한 오염은 줄눈 코팅이나 재시공이 안전합니다.',
        'clean_tip': '줄눈 상태가 양호합니다. 타일 바닥 물기가 고이지 않도록 배수구를 정리하고 환기를 자주 유지해 건조 상태를 지켜주세요.'
    }
}

def load_ai_model():
    global model, model_status, model_error_msg, classes, preprocess
    if not MODEL_PATH.exists():
        model_status = 'not_found'
        model_error_msg = f'모델 파일을 찾을 수 없습니다: {MODEL_PATH.name}'
        print(f'[CLEAN:ON] {model_error_msg}')
        return

    try:
        # Colab에서 저장된 체크포인트 로드
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        
        # 체크포인트 내부 메타데이터 추출
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            loaded_classes = checkpoint.get('classes', ['black_spots', 'clean'])
            img_size = checkpoint.get('image_size', 224)
            mean = checkpoint.get('mean', [0.485, 0.456, 0.406])
            std = checkpoint.get('std', [0.229, 0.224, 0.225])
        elif isinstance(checkpoint, dict):
            # 혹시 state_dict만 직접 저장된 dict인 경우
            state_dict = checkpoint
            loaded_classes = ['black_spots', 'clean']
            img_size = 224
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            raise ValueError('체크포인트 형식이 유효하지 않습니다.')

        classes = list(loaded_classes)

        # ResNet18 생성 및 최종 fc 계층을 클래스 수에 맞게 설정
        net = models.resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
        net.load_state_dict(state_dict)
        net.eval()
        model = net

        # 체크포인트의 image_size, mean, std로 전처리 파이프라인 구성
        target_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        model_status = 'ready'
        model_error_msg = None
        print(f'[CLEAN:ON] 모델 로드 완료 (ResNet18, classes={classes}, size={target_size})')
    except Exception as e:
        model = None
        model_status = 'error'
        model_error_msg = str(e)
        print(f'[CLEAN:ON] 모델 로드 실패: {e}')

# 서버 기동 시 모델 로드
load_ai_model()

@app.get('/')
def home():
    return FileResponse(BASE / 'index.html')

@app.get('/api/status')
def get_status():
    """서버 및 AI 모델 상태 확인 엔드포인트"""
    return {
        'model_status': model_status,
        'model_loaded': model is not None,
        'model_path_exists': MODEL_PATH.exists(),
        'classes': classes if model is not None else [],
        'error': model_error_msg
    }

@app.post('/api/analyze')
async def analyze(image: UploadFile = File(...), material: str = Form(...)):
    # 1. 재질 파라미터 검증
    if material not in CLEANING_GUIDES:
        raise HTTPException(status_code=400, detail='지원하지 않는 재질입니다. (silicone, tile, grout 중 선택)')

    guide_info = CLEANING_GUIDES[material]

    # 2. 이미지 확장자 및 Content-Type 검증
    ct = (image.content_type or '').split(';')[0].strip().lower()
    allowed_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/pjpeg', 'image/x-png', 'application/octet-stream'}
    ext = Path(image.filename or '').suffix.lower()
    if ct not in allowed_types and ext not in {'.jpg', '.jpeg', '.png'}:
        raise HTTPException(status_code=400, detail='JPG 또는 PNG 형식의 이미지 파일만 업로드할 수 있습니다.')

    # 3. 파일 용량 제한 (최대 5MB)
    raw = await image.read(5 * 1024 * 1024 + 1)
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='이미지 파일 크기가 5MB를 초과했습니다.')

    # 4. 이미지 무결성 검증 및 RGB 변환
    try:
        pic = Image.open(BytesIO(raw))
        pic.verify()
        pic = Image.open(BytesIO(raw)).convert('RGB')
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail='유효한 이미지 파일이 아니거나 손상된 파일입니다.')

    # 5. 모델 미탑재 또는 오류 상태일 때: 실제 분석 결과를 위조하지 않고 명확한 데모/오류 모드로 응답
    if model is None or preprocess is None:
        return {
            'mode': 'demo',
            'status': 'demo' if model_status == 'not_found' else 'error',
            'label': None,
            'label_name': '데모 모드 (AI 모델 미적용)',
            'confidence': None,
            'confidence_percent': None,
            'material': material,
            'material_name': guide_info['title'],
            'message': 'AI 학습 모델을 불러올 수 없어 기본 청소 가이드만 제공하는 데모 모드로 작동 중입니다.',
            'error_detail': model_error_msg,
            'guide': {
                'title': guide_info['title'],
                'description': guide_info['description'],
                'steps': guide_info['steps'],
                'material_tip': guide_info['spots_tip'],
                'safety_warning': '안전 수칙: 환기팬 가동 및 창문 개방은 필수이며, 보호장갑과 마스크를 착용하세요. 염소계 표백제(락스)를 산성 세제나 식초, 암모니아와 혼합하면 유독 염소가스가 발생하므로 절대 혼합하지 마세요.'
            },
            'disclaimer': '데모 모드: 실제 사진 분석이 수행되지 않았으며 일반 참고용 가이드입니다.'
        }

    # 6. 실제 AI 모델 추론 수행
    try:
        input_tensor = preprocess(pic).unsqueeze(0)
        with torch.inference_mode():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            score_tensor, idx_tensor = torch.max(probs, dim=0)
            score = float(score_tensor.item())
            idx = int(idx_tensor.item())
            pred_class = classes[idx]

        # 검은 점상 오염 vs 정상 상태 구분
        is_black_spots = (pred_class == 'black_spots')
        if is_black_spots:
            label_name = '검은 점상 오염 패턴 유사'
            material_tip = guide_info['spots_tip']
        else:
            label_name = '특이 오염 패턴 적음 (정상 상태)'
            material_tip = guide_info['clean_tip']

        return {
            'mode': 'model',
            'status': 'success',
            'label': pred_class,
            'label_name': label_name,
            'is_black_spots': is_black_spots,
            # 주의사항 반영: 20장 소규모 데이터이므로 '정확도'가 아닌 '모델 분류 점수'로 제공
            'confidence': round(score, 4),
            'confidence_percent': round(score * 100, 1),
            'material': material,
            'material_name': guide_info['title'],
            'message': 'ResNet18 모델 분석이 완료되었습니다.',
            'guide': {
                'title': guide_info['title'],
                'description': guide_info['description'],
                'steps': guide_info['steps'],
                'material_tip': material_tip,
                'safety_warning': '안전 수칙: 작업 시 충분한 환기와 보호장갑 착용을 권장합니다. 염소계 표백제(락스)는 산성 세제·식초·암모니아와 절대 혼합하지 마세요.'
            },
            'disclaimer': '본 결과는 소규모(총 20장) 데이터셋으로 학습된 ResNet18 모델의 패턴 유사도 점수이며, 실제 정확도나 오염 종류에 대한 확정 진단이 아닙니다. 촬영 환경(조명, 각도, 해상도)에 따라 오차가 발생할 수 있습니다.'
        }
    except Exception as inference_error:
        # 추론 중 예상치 못한 오류 발생 시 위조 결과를 내지 않고 명확한 오류 반환
        return {
            'mode': 'error',
            'status': 'error',
            'label': None,
            'label_name': '분석 오류',
            'confidence': None,
            'confidence_percent': None,
            'material': material,
            'material_name': guide_info['title'],
            'message': f'이미지 분석 중 오류가 발생했습니다: {str(inference_error)}',
            'guide': {
                'title': guide_info['title'],
                'description': guide_info['description'],
                'steps': guide_info['steps'],
                'material_tip': guide_info['spots_tip'],
                'safety_warning': '안전 수칙: 충분한 환기와 보호장갑 착용을 권장합니다. 표백제와 산성 세제 혼합은 절대 금지합니다.'
            },
            'disclaimer': '오류 상태: 모델 추론에 실패하여 기본 청소 가이드만 제공됩니다.'
        }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('server:app', host='127.0.0.1', port=8000, reload=True)
