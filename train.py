from pathlib import Path
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

BASE = Path(__file__).resolve().parent
DATA = BASE / 'dataset'
if not DATA.exists():
    print(f'데이터셋 폴더({DATA})를 찾을 수 없습니다.')
    print('dataset/black_spots 및 dataset/clean 폴더를 생성하고 이미지를 추가한 뒤 다시 실행해 주세요.')
    raise SystemExit(1)
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
dataset = datasets.ImageFolder(DATA, transform=transform)
assert dataset.classes == ['black_spots','clean'], f'폴더 이름 확인: {dataset.classes}'
assert len(dataset) >= 20, '최소 20장 이상의 예제 사진이 필요합니다 (성능 보장 아님).'
seed = torch.Generator().manual_seed(42)
train_len = int(len(dataset)*0.8)
train_set, val_set = random_split(dataset, [train_len,len(dataset)-train_len], generator=seed)
train_loader = DataLoader(train_set,batch_size=8,shuffle=True)
val_loader = DataLoader(val_set,batch_size=8)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad=False
model.fc = nn.Linear(model.fc.in_features,2)
optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()
best_acc=-1
for epoch in range(5):
    model.train()
    for x,y in train_loader:
        optimizer.zero_grad();loss=loss_fn(model(x),y);loss.backward();optimizer.step()
    model.eval();correct=total=0
    with torch.no_grad():
        for x,y in val_loader:
            pred=model(x).argmax(1);correct+=(pred==y).sum().item();total+=len(y)
    acc=correct/max(total,1)
    print(f'Epoch {epoch+1}/5 - 검증 정확도 {acc:.1%} (표본 {total}장)')
    if acc>best_acc: best_acc=acc;torch.save(model.state_dict(),BASE/'bathroom_model.pth')
print('저장 완료: bathroom_model.pth / 소규모 검증 점수는 신뢰성 보장하지 않음')
