# test.py
"""
快速自验：确认自定义 AMP 封装确已生效
运行：
    CUDA_VISIBLE_DEVICES=0 python test.py
如全部断言通过且打印信息合理，即说明混合精度正常工作。
"""

import torch
from LGDiffusion.Functions import AMP   # 你的 AMP 封装

# ──────────────────────────────────────────────────────────────── #
# 1) 准备一个极简 CNN                                            #
# ──────────────────────────────────────────────────────────────── #
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ──────────────────────────────────────────────────────────────── #
# 2) 单步训练 & 统计函数                                         #
# ──────────────────────────────────────────────────────────────── #
def run_step(fp16: bool):
    torch.cuda.reset_peak_memory_stats()

    model = SimpleCNN().cuda()
    amp    = AMP(model, fp16)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)

    x      = torch.randn(8, 3, 64, 64, device="cuda")
    target = torch.randn_like(x)

    with amp.autocast():                       # 关键：前向应进入 autocast
        out  = model(x)
        loss = torch.nn.functional.mse_loss(out, target)

    amp.backward(loss, optim, accumulate_grad=1)
    amp.step(optim)
    optim.zero_grad()

    dtype_out  = out.dtype
    scaler_val = amp.scaler.get_scale()        # 对于 fp32 模式恒为 1
    peak_mem   = torch.cuda.max_memory_allocated() / 2**20  # MiB

    return dtype_out, scaler_val, peak_mem

# ──────────────────────────────────────────────────────────────── #
# 3) 对比 FP32 vs FP16                                            #
# ──────────────────────────────────────────────────────────────── #
dtype32, scale32, mem32 = run_step(fp16=False)
dtype16, scale16, mem16 = run_step(fp16=True)

print(f"\nFP32 → dtype={dtype32}, scaler={scale32}, peak_mem={mem32:.1f} MiB")
print(f"FP16 → dtype={dtype16}, scaler={scale16}, peak_mem={mem16:.1f} MiB")

# ─── 断言：全部满足则 AMP 工作正常 ──────────────────────────────── #
assert dtype16 == torch.float16,  "前向输出不是 FP16，AMP 未生效"
assert dtype32 == torch.float32,  "FP32 路径输出异常"
assert scale16 > 1,              "GradScaler 未启用或未更新"
assert mem16 < mem32 * 0.8,      "显存未明显下降，怀疑 AMP 未生效"

print("\n✓ 所有检测通过：混合精度机制已正常工作！")
