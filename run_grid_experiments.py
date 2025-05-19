import itertools
import subprocess

# 고정 인자
fixed_args = {
    "--project": "household",
    "--compound_coef": "0",
    "--load_weights": "./weights/efficientdet-d0.pth",
    "--num_workers": "4",
    "--head_only": "False",
    "--optim": "adamw"
}

# 실험할 인자 조합
batch_sizes = [8, 16, 32]
lrs = [1e-3, 1e-4, 5e-4]

# num_epochs_list = [100, 200]
num_epochs_list = [100]

# 실험 번호로 식별
experiment_id = 1

# 모든 조합에 대해 실행
for batch_size, lr, num_epochs in itertools.product(batch_sizes, lrs, num_epochs_list):
    # 고정 및 가변 인자 조합
    cmd = ["python", "train.py"]
    for k, v in fixed_args.items():
        cmd += [k, v]
    cmd += [
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--num_epochs", str(num_epochs),
        "--log_path", f"logs/exp{experiment_id}",
        "--saved_path", f"checkpoints/exp{experiment_id}"
    ]

    print(f"🚀 실행 중: Experiment {experiment_id} (bs={batch_size}, lr={lr}, epochs={num_epochs})")
    subprocess.run(cmd)
    experiment_id += 1

"""
# 만들어 지는 커맨드 예시

python train.py \
  --project household \
  --compound_coef 0 \
  --num_workers 4 \
  --head_only False \
  --optim adamw \
  --batch_size 8 \
  --lr 0.0001 \
  --num_epochs 100 \
  --log_path logs/exp1 \
  --saved_path checkpoints/exp1

"""

