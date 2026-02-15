# Gym PushT Experiments

## Requires python=3.10

## Install required packages and ffmpeg if it's not already installed
```bash
pip install -r requirements.txt
sudo apt-get install ffmpeg
```

## Train script
```bash
accelerate launch --num_machines=1 --num_processes=8 --mixed_precision=no --dynamo_backend=no $(which lerobot-train) \
    --output_dir=./my_diffusion_pusht \
    --config_path=./my_diffusion_pusht/checkpoints/last/pretrained_model/train_config.json \
    --policy.type=diffusion \
    --dataset.repo_id=lerobot/pusht \
    --seed=100000 \
    --env.type=pusht \
    --batch_size=64 \
    --steps=200000 \
    --eval_freq=25000 \
    --save_freq=25000 \
    --wandb.enable=true \
    --policy.push_to_hub=false \
    --resume=true
```

## Eval script
```bash
python render.py
```

## Example

(Click to watch the video)

[![PushT example video](https://img.youtube.com/vi/JLRBWaBbMdk/0.jpg)](https://youtu.be/JLRBWaBbMdk)
