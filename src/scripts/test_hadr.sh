#!/usr/bin/env sh
default_cfg_path=./scripts/default_config.yaml
cfg_paths=./scripts/test_lidf.yaml

python main.py \
    --default_cfg_path $default_cfg_path \
    --cfg_paths $cfg_paths
