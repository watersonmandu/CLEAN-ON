import argparse
import json
from pathlib import Path
import torch
from torchvision import models
import onnx

def convert(pth_path: Path, onnx_path: Path, meta_json_path: Path = None):
    print(f"[ONNX 변환 시작] PyTorch 모델 로드 중: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        classes = list(checkpoint.get('classes', ['black_spots', 'clean']))
        img_size = checkpoint.get('image_size', 224)
        mean = list(checkpoint.get('mean', [0.485, 0.456, 0.406]))
        std = list(checkpoint.get('std', [0.229, 0.224, 0.225]))
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        classes = ['black_spots', 'clean']
        img_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"지원하지 않는 체크포인트 형식: {type(checkpoint)}")

    if isinstance(img_size, (list, tuple)):
        img_h, img_w = img_size[0], img_size[1]
    else:
        img_h = img_w = int(img_size)

    print(f" - 클래스: {classes}")
    print(f" - 입력 크기: ({img_h}, {img_w})")
    print(f" - 정규화 Mean: {mean}, Std: {std}")

    net = models.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(classes))
    net.load_state_dict(state_dict)
    net.eval()

    dummy_input = torch.randn(1, 3, img_h, img_w, dtype=torch.float32)

    print(f"[ONNX Export] ONNX 파일로 내보내기: {onnx_path}")
    export_kwargs = {
        'export_params': True,
        'opset_version': 14,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    }
    # PyTorch 2.5+ 호환: dynamo=False로 TorchScript 기반 export 보장
    try:
        torch.onnx.export(net, dummy_input, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(net, dummy_input, str(onnx_path), **export_kwargs)

    # ONNX 모델 메타데이터 주입
    onnx_model = onnx.load(str(onnx_path))
    meta_dict = {
        'classes': json.dumps(classes),
        'image_size': json.dumps([img_h, img_w]),
        'mean': json.dumps(mean),
        'std': json.dumps(std),
        'architecture': 'resnet18'
    }
    for k, v in meta_dict.items():
        meta = onnx_model.metadata_props.add()
        meta.key = k
        meta.value = v
    onnx.save(onnx_model, str(onnx_path))
    print(f"[ONNX 메타데이터 저장 완료]")

    # 동반 JSON 메타데이터 저장 (선택/권장)
    if meta_json_path is None:
        meta_json_path = onnx_path.with_name(onnx_path.stem + '_meta.json')
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'classes': classes,
            'image_size': [img_h, img_w],
            'mean': mean,
            'std': std,
            'architecture': 'resnet18'
        }, f, indent=2, ensure_ascii=False)
    print(f"[JSON 메타데이터 저장 완료]: {meta_json_path}")

    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[완료] 생성된 ONNX 모델 크기: {file_size_mb:.2f} MB")
    print(f" - GitHub 업로드 허용치(100MB 이하) 충족: {file_size_mb < 100}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet18 모델을 ONNX 형식으로 변환합니다.')
    parser.add_argument('--pth', type=Path, default=Path('bathroom_model.pth'), help='입력 .pth 모델 파일 경로')
    parser.add_argument('--onnx', type=Path, default=Path('bathroom_model.onnx'), help='출력 .onnx 파일 경로')
    args = parser.parse_args()

    convert(args.pth, args.onnx)
