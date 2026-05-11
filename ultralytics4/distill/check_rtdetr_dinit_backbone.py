import torch
from ultralytics import RTDETR

# 1. 加载 COCO 原始模型
model_coco = RTDETR("rtdetr-l.pt").model
state_coco = model_coco.state_dict()

# 2. 加载蒸馏 checkpoint
ckpt = torch.load(
    "EFM5/fusion/runs/rtdetr_distill_debug/rtdetr_distilled.pth",
    map_location="cpu",
    weights_only=True
)
student_state = ckpt["student_state"]

# 3. 加载你生成的初始化权重
state_dinit = torch.load(
    "ultralytics4/distill/runs/exp_rtdetr_distill/rtdetr-l-coco-dinit.pt",
    map_location="cpu"
)

# 4. 随便选一个 backbone 参数
key = "model.0.stem1.conv.weight"

w_coco = state_coco[key]
w_distill = student_state[key]
w_dinit = state_dinit[key]

# 5. 计算差异
def diff(a, b):
    return (a - b).abs().mean().item()

print("COCO vs 蒸馏:", diff(w_coco, w_distill))
print("COCO vs DInit:", diff(w_coco, w_dinit))
print("蒸馏 vs DInit:", diff(w_distill, w_dinit))
