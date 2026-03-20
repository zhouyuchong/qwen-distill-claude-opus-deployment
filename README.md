# qwen-distill-claude-opus-deployment

Scripts for deploying [Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled)

## Create Conda env
```
conda create -p /path-to-conda-env python=3.11
conda activate /path-to-conda-envs
```

## Conda Environment Dependencies
install the requirements.  
- run `pip install -r requirements.txt`
- For all details of my configuration, see [my-conda-env-dump-details.md](./my-conda-env-dump-details.md)

## Installation
Download [weight files](https://huggingface.co/Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled/discussions) to `/path-to-weight`, from which you see:
```
- /path-to-weight
  - .gitattributes
  - README.md
  - chat_template.jinja
  - config.json
  - model.safetensors-00001-of-00011.safetensors
  - ...
```

## Start vllm and proxy
Start a command tab.
```
conda activate /path-to-conda-envs
bash start_vllm.sh
```

Start another command tab.
```
conda activate /path-to-conda-envs
bash start_proxy.sh
```

vi ~/.claude/settings.json(fill in your IP).
```
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://{your ip}:8800",
    "ANTHROPIC_AUTH_TOKEN": "sk-placeholder",
    "ANTHROPIC_MODEL": "Qwen3.5-27B-Opus"
  },
}
```