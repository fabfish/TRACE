# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
import torch_npu
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
# from transformers.integrations.deepspeed import HfDeepSpeedConfig
# check if HfDeepSpeedConfig is available
try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    

    if disable_dropout:
        model_config.dropout = 0.0

    # --- FIX STARTS HERE ---
    # Modify the config object BEFORE creating the model.
    # Llama models use eos_token_id but sometimes generation configs expect end_token_id.
    model_config.end_token_id = tokenizer.eos_token_id
    
    # Set pad_token_id to the same value as eos_token_id.
    # This ensures it's an integer when GenerationConfig is created.
    model_config.pad_token_id = tokenizer.pad_token_id if model_config.pad_token_id is not None else None
    # --- FIX ENDS HERE ---

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    # print(model_config)

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True,
        # torch_dtype=torch.float16,
        )

    # # llama use eos_token_id but not end_token_id
    # model.config.end_token_id = tokenizer.eos_token_id
    # # compatible with OPT and llama2
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8


    return model
