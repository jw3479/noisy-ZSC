import wandb
import yaml

with open('tests/other_play_ippo.yml', 'r') as stream:
    try:
        sweep_config=yaml.safe_load(stream)
        print(sweep_config)
    except yaml.YAMLError as exc:
        print(exc)

sweep_id = wandb.sweep(sweep=sweep_config)
wandb.agent(sweep_id)
