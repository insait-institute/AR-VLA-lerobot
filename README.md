# ART policy (external package)
This repo contains an external policy package for the paper [AR-VLA: Autoregressive Action Expert for Vision-Language-Action Models](https://arxiv.org/abs/2603.10126).

This is an example of how to implement an Auto Regressive Transformer (ART) policy as an external LeRobot package.

For specialist benchmark results (and ACT/DP baselines), use the `lerobot-v2` branch. Reproduction instructions are in the `lerobot-v2` README, and training logs are available on [wandb](https://api.wandb.ai/links/hu2240877635/rtm8myee).

The generalist AR-VLA model pretrained on BridgeV2 (SIMPLER + real-world WidowX250) is available on [huggingface](https://hf.co/collections/you2who/ar-vla).


# Usage
## Install LeRobot v0.4.2

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout v0.4.2
pip install -e .
cd ..
```

you can also consider installing the 0.4.2 version lerobot directly from pip, as we do not change any lerobot code, but we recommend installing from source for easier debugging and modification.

## Install this policy package
Inside this repo folder

```bash
pip install -e .
```

## (For the custom PushT2 task) Install the environment

We overwrite the `gym-pusht` package to add the custom PushT2 environment while keeping the original PushT-v0.

```bash
cd /path/to/new_pusht_folder
git clone https://github.com/utomm/gym-pusht.git
cd gym-pusht
pip install -e .
```

## Train with ART

You must load the external policy package at runtime with `--policy.discover_packages_path`.

Example (only include non-default args you want to override):

```bash
lerobot-train \
  --policy.discover_packages_path=lerobot_policy_art \
  --policy.type=art \
  --dataset.repo_id=you2who/pusht2-teleop-v30 \
  --env.type=pusht \
  --env.task=PushT2-v0 \
  --steps=200000 \
  --batch_size=64
```

Notes:
- Parameters that match defaults in `ARTConfig` can be omitted.
- If you override action chunking or history settings, keep `history_length + chunk_size` aligned with your dataset window.
