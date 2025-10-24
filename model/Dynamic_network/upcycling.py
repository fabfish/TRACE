from copy import deepcopy
import torch
import torch.nn.functional as F
import torch_npu
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from deepspeed.utils import safe_get_full_grad
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds # to be continued
from transformers import GenerationConfig
import json
import os
import math
import types

import deepspeed

generation_config = GenerationConfig(
    temperature=0.1,
    do_sample=True,
    num_return_sequences=1
)

# --- Helper Classes for MoE components ---

class Expert(nn.Module):
    """A standard MLP expert."""
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Router(nn.Module):
    """Router module for dispatching tokens to experts."""
    def __init__(self, config, num_experts):
        super().__init__()
        self.top_k = config.num_activated_experts
        # Initialize the router classifier with the full number of experts
        self.classifier = nn.Linear(config.hidden_size, num_experts) 

    def forward(self, hidden_states: torch.Tensor):
        # Input shape: [batch_size, seq_len, hidden_dim] -> [num_tokens, hidden_dim]
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        logits = self.classifier(hidden_states)
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        
        # Get top-k experts and their weights
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize the top-k weights
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


def moe_forward(self, x):
    """
    New forward pass for the MLP layer, turning it into an MoE layer.
    This function will be monkey-patched into each LlamaMLP instance.
    `self` refers to the LlamaMLP layer instance.
    """
    # 1. Shared Expert (Original FFN)
    # print("x is", x)
    shared_expert_output = self.original_forward(x)

    # 2. Scientific Experts
    if len(self.scientific_experts) > 0:
        routing_weights, selected_experts = self.router(x)
        final_expert_output = torch.zeros_like(shared_expert_output)
        
        # Flatten input for expert processing
        flat_x = x.view(-1, x.shape[-1])
        
        # Get flattened routing info
        flat_routing_weights = routing_weights.view(-1, routing_weights.shape[-1])
        flat_selected_experts = selected_experts.view(-1, selected_experts.shape[-1])

        # Iterate over each token and its selected experts
        for i in range(flat_x.shape[0]):
            token_output = 0
            for j in range(self.router.top_k):
                expert_idx = flat_selected_experts[i, j].item()
                expert_layer = self.scientific_experts[expert_idx]
                weight = flat_routing_weights[i, j]
                token_output += weight * expert_layer(flat_x[i].unsqueeze(0))
            final_expert_output.view(-1, x.shape[-1])[i] = token_output
            
        return shared_expert_output + final_expert_output
    else:
        return shared_expert_output
    
# New function to convert the model to an MoE model
def convert_upcycle_model(model, args, num_tasks=0, incremental=False):
    """
    Convert the model to an Upcycling MoE model.

    - If incremental == False: build model with num_tasks * num_experts_per_task experts (same as before).
    - If incremental == True: append num_tasks * num_experts_per_task new experts to existing MoE
      (if model already has experts), otherwise initialize and append.

    参数:
    - model: HF model wrapper (expected to have model.model.layers)
    - args: 包含 num_experts_per_task, num_activated_experts, device 等
    - num_tasks: 当 incremental=False 表示总任务数；当 incremental=True 表示新增任务数
    - incremental: 是否以增量方式追加专家（True）还是重建为总专家数（False）
    """
    num_experts = int(getattr(args, 'num_experts_per_task', 8))
    device = getattr(args, 'device', torch.device("npu"))
    # make sure device is torch.device or string acceptable by .to()
    if isinstance(device, str):
        device_str = device
    else:
        device_str = device

    if incremental:
        num_new_experts = int(num_tasks) * num_experts
        if num_new_experts <= 0:
            print_rank_0(f"[Upcycle] incremental=True but num_new_experts=={num_new_experts}, nothing to do.", getattr(args, 'global_rank', 0))
            return model
        print_rank_0(f"⚙️ Incrementally adding {num_new_experts} experts per layer.", getattr(args, 'global_rank', 0))

        for li, layer in enumerate(model.model.layers):
            if li % 2 != 0:
                continue

            mlp = layer.mlp
            # Ensure existing MoE structures present; if not, initialize as base
            if not hasattr(mlp, "scientific_experts") or len(mlp.scientific_experts) == 0:
                # initialize MLP to MoE baseline first (no experts, but router present)
                mlp.original_forward = mlp.forward
                mlp.scientific_experts = nn.ModuleList([])
                if not hasattr(model.model.config, 'num_activated_experts'):
                    model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))
                mlp.router = Router(model.model.config, 0)

            # Use main FFN weights as source for new experts
            W_g = mlp.gate_proj.weight.data  # [H, h]
            W_u = mlp.up_proj.weight.data    # [H, h]
            W_d = mlp.down_proj.weight.data  # [h, H]

            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            # intermediate size for each expert (consistent with how experts were derived)
            new_intermediate_size = H // num_experts

            old_total = len(mlp.scientific_experts)
            for idx in range(num_new_experts):
                # build new expert
                new_expert = Expert(model.model.config, new_intermediate_size)
                new_expert.to(device_str)
                expert_idx_in_task = (old_total + idx) % num_experts
                start_col = expert_idx_in_task * new_intermediate_size
                end_col = (expert_idx_in_task + 1) * new_intermediate_size

                # copy slices from shared FFN (same logic as upcycle_one_task)
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

                mlp.scientific_experts.append(new_expert)

            # expand router classifier
            old_classifier = mlp.router.classifier
            old_out, old_in = old_classifier.weight.data.shape if hasattr(old_classifier, 'weight') else (0, H)
            new_out = old_out + num_new_experts
            new_router = nn.Linear(old_in, new_out).to(device_str)
            if old_out > 0:
                # copy old weights and bias into prefix
                new_router.weight.data[:old_out, :].copy_(old_classifier.weight.data)
                if old_classifier.bias is not None:
                    new_router.bias.data[:old_out].copy_(old_classifier.bias.data)
            mlp.router.classifier = new_router

        return model

    # non-incremental: build total experts = num_tasks * num_experts (legacy behavior)
    num_total_experts = int(num_tasks) * num_experts
    print_rank_0(f"⚙️ Converting to Upcycling model with {num_total_experts} total experts for {num_tasks} tasks.", getattr(args, 'global_rank', 0))

    for li, layer in enumerate(model.model.layers):
        if li % 2 != 0:
            continue
        print_rank_0(f" Converting layer {li} to MoE layer.", getattr(args, 'global_rank', 0))

        mlp = layer.mlp

        # store original forward
        mlp.original_forward = mlp.forward
        mlp.scientific_experts = nn.ModuleList([])

        if not hasattr(model.model.config, 'num_activated_experts'):
            model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))

        # create router sized for total experts
        mlp.router = Router(model.model.config, num_total_experts)

        h = mlp.gate_proj.in_features
        H = mlp.gate_proj.out_features
        W_g = mlp.gate_proj.weight.data
        W_u = mlp.up_proj.weight.data
        W_d = mlp.down_proj.weight.data
        new_intermediate_size = H // num_experts

        for ei in range(num_total_experts):
            new_expert = Expert(model.model.config, new_intermediate_size)
            new_expert.to(device_str)

            task_id = ei // num_experts
            expert_idx_in_task = ei % num_experts

            start_col = expert_idx_in_task * new_intermediate_size
            end_col = (expert_idx_in_task + 1) * new_intermediate_size

            new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
            new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
            new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

            mlp.scientific_experts.append(new_expert)

        # patch forward
        layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)

    return model


class Upcycle(CL_Base_Model):
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args,):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        
        self.current_task_id = 0
        self.task2expert_range = {}
        # Assumes args for MoE are passed from main.py
        self.num_experts_per_task = getattr(args, 'num_experts_per_task', 8)
        self.num_activated_experts = getattr(args, 'num_activated_experts', 2)
        # Upcycle control: interval and explicit task names
        self.upcycle_interval = getattr(args, 'upcycle_interval', 4)
        raw_names = getattr(args, 'upcycle_task_names', [])
        if isinstance(raw_names, str):
            names = raw_names.split(',') if raw_names else []
        else:
            names = raw_names
        self.upcycle_task_names = set([n.strip() for n in names if n and n.strip()])

        print_rank_0(f"[MoE-Upcycle] Initializing model with {self.num_experts_per_task} experts per task and {self.num_activated_experts} activated experts.", self.args.global_rank)

        # Before DeepSpeed initialization, modify the model architecture
        
        # for training
        # self._prepare_model_for_upcycling()

        if not hasattr(self.args, 'device'):
            self.args.device = torch.device("npu")


    def _prepare_model_for_upcycling(self):
        """
        Injects MoE components (expert list, router) into each MLP layer of the base model.
        Also patches the forward method.
        """
        for i, layer in enumerate(self.model.model.layers):
            if i % 2 == 0:
                mlp = layer.mlp
                
                # Store original FFN forward method
                # mlp.original_forward = types.MethodType(mlp.forward, mlp)
                mlp.original_forward = mlp.forward  # 不要用 MethodType
                print_rank_0(f" original_forward is {mlp.original_forward}", self.args.global_rank)

                # # test the method
                # test_input = torch.randn(2, 4, mlp.gate_proj.in_features).to("npu")
                # test_output = mlp.original_forward(test_input)
                # print_rank_0(f"Test original_forward output shape: {test_output.shape}", self.args.global_rank)
                
                # Add MoE components
                mlp.scientific_experts = nn.ModuleList([])
                self.model.model.config.num_activated_experts = self.num_activated_experts
                mlp.router = Router(self.model.model.config)
                
                # Monkey-patch the forward method
                layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
                # layer.mlp.forward = moe_forward
                print_rank_0(f"Patched forward method for layer {i}", self.args.global_rank)
            else:
                print_rank_0(f"Skipping layer {i} for MoE conversion", self.args.global_rank)


    def train_continual(self):
         for i_task, task in enumerate(self.train_task_list):
            self.current_task_id = i_task
            print_rank_0(f"[MoE-Upcycle] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

#            # 1. For a new task, upcycle the model by adding new experts and expanding the router.
#            self.upcycle_one_task(task, i_task)

            # Decide whether to upcycle for this task:
            # - upcycle if index modulo interval == 0
            # - OR if task name listed in upcycle_task_names
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True

            if do_upcycle:
                # 1. For a new task, upcycle the model by adding new experts and expanding the router.
                self.upcycle_one_task(task, i_task)
            else:
                print_rank_0(f"[MoE-Upcycle] Skipping upcycle for task-{i_task}: {task}", self.args.global_rank)

            # 2. Train the model on the current task. Only newly added parameters will be updated.
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # 3. Save the model state after training on the task.
            self.save_model(i_task)

    def upcycle_one_task(self, task, i_task):
        """
        Adds and initializes a new set of experts for the current task based on the paper's
        "Scientific Expert Split" methodology.
        """
        if i_task in self.task2expert_range:
            print_rank_0(f"Task {i_task} already has experts. Skipping upcycling.", self.args.global_rank)
            return

        print_rank_0(f"Upcycling model for new task: {task} (Task ID: {i_task})", self.args.global_rank)
        
        # layer = self.model.model.layers[0]
        if hasattr(self.model.model.layers[0].mlp, "scientific_experts") and len(self.model.model.layers[0].mlp.scientific_experts) > 0:
            print_rank_0(f"Current number of experts per layer before upcycling: {len(self.model.model.layers[0].mlp.scientific_experts)}", self.args.global_rank)

            # total_experts_before = sum(len(layer.mlp.scientific_experts) for layer in self.model.model.layers) // len(self.model.model.layers)
            # some layer may not have experts
            total_experts_before = len(self.model.model.layers[0].mlp.scientific_experts)

        else:
            total_experts_before = 0
            print_rank_0(f"No existing experts found. Starting fresh.", self.args.global_rank)

            for i, layer in enumerate(self.model.model.layers):
                if i % 2 != 0:
                    continue
                    pass
                else:
                    print(f" Converting layer {i} to MoE layer.")

                mlp = layer.mlp
                
                # Store original FFN forward method
                # mlp.original_forward = types.MethodType(mlp.forward, mlp)
                mlp.original_forward = mlp.forward  # 不要用 MethodType
                print_rank_0(f" original_forward is {mlp.original_forward}", self.args.global_rank)

                # # test the method
                # test_input = torch.randn(2, 4, mlp.gate_proj.in_features).to("npu")
                # test_output = mlp.original_forward(test_input)
                # print_rank_0(f"Test original_forward output shape: {test_output.shape}", self.args.global_rank)
                
                # Add MoE components
                mlp.scientific_experts = nn.ModuleList([])
                self.model.model.config.num_activated_experts = self.num_activated_experts
                # mlp.router = Router(self.model.model.config)
                mlp.router = Router(self.model.model.config, 0)
                
                # Monkey-patch the forward method
                layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
                # layer.mlp.forward = moe_forward
                print_rank_0(f"Patched forward method for layer {i}", self.args.global_rank)

        start_expert_idx = total_experts_before
        end_expert_idx = start_expert_idx + self.num_experts_per_task
        self.task2expert_range[i_task] = range(start_expert_idx, end_expert_idx)

        # for layer in self.model.model.layers:
        for i, layer in enumerate(self.model.model.layers):
            if i % 2 != 0:
                continue
                pass
            else:
                print(f" Appending experts to layer {i}.")

            mlp = layer.mlp
            
            # Get the weights from the shared expert (original FFN)
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            
            # Calculate the intermediate size for the new, smaller experts
            new_intermediate_size = H // self.num_experts_per_task # // 64
            # print(" new_intermediate_size is", new_intermediate_size)
            print_rank_0(f" Intermediate size for new experts: {new_intermediate_size}", self.args.global_rank)
        

            # Create and initialize new experts for the current task
            for i in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(self.args.device)
                
                # Split weights and initialize ("Scientific Expert Split")
                start_col = i * new_intermediate_size
                end_col = (i + 1) * new_intermediate_size
                
                # new_expert.gate_proj.weight.data = W_g[:, start_col:end_col].clone()
                # new_expert.up_proj.weight.data = W_u[:, start_col:end_col].clone()
                # new_expert.down_proj.weight.data = W_d[start_col:end_col, :].clone()

                # Linear(in, out) -> weight shape [out, in]
                # gate_proj: [H, h], up_proj: [H, h], down_proj: [h, H]
                # 你要切的是 out 维度（第0维），不是 in 维度（第1维）
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                
                mlp.scientific_experts.append(new_expert)
            
            # Expand the router's classifier layer
            num_total_experts = len(mlp.scientific_experts)
            new_router_classifier = nn.Linear(h, num_total_experts, device=self.args.device)
            # Copy old weights
            new_router_classifier.weight.data[:total_experts_before, :] = mlp.router.classifier.weight.data
            # Keep new weights at default initialization
            mlp.router.classifier = new_router_classifier

        print_rank_0(f"Added {self.num_experts_per_task} new experts. Total experts per layer: {len(self.model.model.layers[0].mlp.scientific_experts)}", self.args.global_rank)


    # def freeze_non_current_task_params(self, i_task):
    #     """
    #     Freezes all parameters except for the experts and router weights for the current task.
    #     """
    #     for param in self.model.parameters():
    #         param.requires_grad = False
            
    #     expert_range_to_train = self.task2expert_range.get(i_task)
    #     if expert_range_to_train is None:
    #         return

    #     print_rank_0(f"Unfreezing experts in range {expert_range_to_train} for task {i_task}", self.args.global_rank)

    #     for layer in self.model.model.layers:
    #         # Unfreeze current task's experts
    #         for expert_idx in expert_range_to_train:
    #             for param in layer.mlp.scientific_experts[expert_idx].parameters():
    #                 param.requires_grad = True
            
    #         # Unfreeze corresponding router weights

    #         # for param in layer.mlp.router.classifier.parameters():
    #         #     param.requires_grad = True
    #         #     # if not param.requires_grad:
    #         #     #     continue
    #         #     # Only unfreeze the output neurons for the new experts
    #         #     grad_mask = torch.zeros_like(param.data)
    #         #     grad_mask[expert_range_to_train.start:expert_range_to_train.stop, :] = 1
                
    #         #     # Use a hook to zero out gradients for frozen parts
    #         #     param.register_hook(lambda grad: grad * grad_mask)

    #         for param in layer.mlp.router.classifier.parameters():
    #             param.requires_grad = True
    #             grad_mask = torch.zeros_like(param.data)
    #             if grad_mask.dim() == 2:
    #                 # 处理权重 [out, in]
    #                 grad_mask[expert_range_to_train.start:expert_range_to_train.stop, :] = 1
    #             elif grad_mask.dim() == 1:
    #                 # 处理偏置 [out]
    #                 grad_mask[expert_range_to_train.start:expert_range_to_train.stop] = 1
    #             else:
    #                 raise RuntimeError(f"Unexpected param shape: {grad_mask.shape}")

    #             # 用 hook 屏蔽梯度
    #             param.register_hook(lambda grad, mask=grad_mask: grad * mask)
            
    #         # # Make sure the router layer itself is trainable
    #         # for param in layer.mlp.router.classifier.parameters():
    #         #     param.requires_grad = True

    def freeze_non_current_task_params(self, i_task):
        """
        只冻结 MLP 层的 expert/router 以外的参数，主干参数保持可训练。
        """
        # 1. 先全部不动
        # 2. 只冻结 MLP 层的 expert/router 以外的参数
        for layer in self.model.model.layers:
            mlp = layer.mlp
            # 先冻结所有 expert 和 router
            for expert in getattr(mlp, "scientific_experts", []):
                for param in expert.parameters():
                    param.requires_grad = False
            if hasattr(mlp, "router"):
                for param in mlp.router.parameters():
                    param.requires_grad = False
            # 冻结 MLP 主干参数
            for name, param in mlp.named_parameters():
                if not name.startswith("scientific_experts") and not name.startswith("router"):
                    param.requires_grad = False

        # 只解冻当前 task 的 expert 和 router
        expert_range_to_train = self.task2expert_range.get(i_task)
        if expert_range_to_train is None:
            return

        print_rank_0(f"Unfreezing experts in range {expert_range_to_train} for task {i_task}", self.args.global_rank)
        # for layer in self.model.model.layers:
        for i, layer in enumerate(self.model.model.layers):
            if i % 2 != 0:
                continue
                pass

            for expert_idx in expert_range_to_train:
                for param in layer.mlp.scientific_experts[expert_idx].parameters():
                    param.requires_grad = True
            for param in layer.mlp.router.classifier.parameters():
                param.requires_grad = True
                grad_mask = torch.zeros_like(param.data)
                if grad_mask.dim() == 2:
                    grad_mask[expert_range_to_train.start:expert_range_to_train.stop, :] = 1
                elif grad_mask.dim() == 1:
                    grad_mask[expert_range_to_train.start:expert_range_to_train.stop] = 1
                else:
                    raise RuntimeError(f"Unexpected param shape: {grad_mask.shape}")
                param.register_hook(lambda grad, mask=grad_mask: grad * mask)


    def train_one_task(self, task, i_task, epochs):
        dataloader_train = self.train_task_list[task]
        device = self.args.device
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        # Freeze all params except those for the current task
        self.freeze_non_current_task_params(i_task)

        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader_train):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Task-{i_task} Epoch-{epoch} Step-{step} loss={loss.item():.4f}")
                
                self.model.backward(loss)
                self.model.step()
                
    def evaluate(self, round, infer_task_id, task):
        self.evaluate_one_task(round, infer_task_id, task)
        
    def evaluate_one_task(self, round, infer_task_id, task):
        device = self.args.device
        infer_dataloader = self.test_task_list[task]
        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            label_sequences = []
            model.eval()

            for step, batch in enumerate(infer_dataloader):
                ground_truths_ids = self.tokenizer(batch['gts'], 
                                                   truncation=True,
                                                   max_length=self.args.max_ans_len,
                                                   add_special_tokens=False,
                                                   padding='max_length',
                                                   return_tensors='pt')['input_ids'].to(device)
                del batch['gts']
                del batch['sources']
                batch = to_device(batch, device)
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)

                with torch.no_grad():
                    generate_ids = model.generate(input_ids=batch['input_ids'],
                                                  attention_mask=batch['attention_mask'],
                                                  max_new_tokens=self.args.max_ans_len,
                                                  bos_token_id=self.tokenizer.bos_token_id,
                                                  eos_token_id=self.tokenizer.eos_token_id,
                                                  pad_token_id=self.tokenizer.unk_token_id,
                                                  generation_config=generation_config,
                                                  use_cache=False)

                # Gather results from all distributed processes
                gathered_ids, max_seq_len = self.dist_results_gather(generate_ids, self.tokenizer.eos_token_id)
                gathered_labels, max_label_len = self.dist_results_gather(ground_truths_ids, self.tokenizer.eos_token_id)

                if self.args.global_rank <= 0:
                    input_len = batch['input_ids'].shape[1]
                    # Decode source, prediction, and labels
                    sou_sequences = self.tokenizer.batch_decode(gathered_ids[:, :input_len], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pre_sequences = self.tokenizer.batch_decode(gathered_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    lab_sequences = self.tokenizer.batch_decode(gathered_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                    predicted_sequences.extend(pre_sequences)
                    sources_sequences.extend(sou_sequences)
                    label_sequences.extend(lab_sequences)

            return sources_sequences, predicted_sequences, label_sequences

        def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                   ground_truths: list, round: int, i_task: int, task: str):
            df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                  'labels': ground_truths}
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)

            with open(os.path.join(self.args.output_dir, f"results-{round}-{i_task}-{task}.json"), "w+", encoding='utf-8') as file:
                json.dump(df, file, ensure_ascii=False, indent=4)

        # Inference
        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(self.model, infer_dataloader)

        # Calculate evaluation metrics
        if self.args.global_rank <= 0:
            task_evaluators = {
                "ScienceQA": eval_ScienceQA.eval,
                "MeetingBank": eval_MeetingBank.eval,
                "C-STANCE": eval_CStance.eval,
                "Papyrus-f": eval_PapyrusF.eval,
                "Py150": eval_Py150.eval,
                "FOMC": eval_FOMC.eval,
                "NumGLUE-cm": eval_NumGLUE_cm.eval,
                "NumGLUE-ds": eval_NumGLUE_ds.eval,
            }
            eval_fn = task_evaluators.get(task, lambda p, g: {})
            evaluation_result = eval_fn(predicted_sequences, ground_truths)
            
            print_rank_0(f"Evaluation result for {task}: {evaluation_result}", self.args.global_rank)
            print_rank_0("***** Saving inference results *****", self.args.global_rank)
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, infer_task_id, task)