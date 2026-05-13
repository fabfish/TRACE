"""
Upcycling MoE Implementation for Prototype
Adapted from the main TRACE implementation for continual learning.

Includes comprehensive Metric Evaluation Module for expanded experts:
- CKA (Centered Kernel Alignment) for representational similarity
- Flatness/Sharpness metrics for landscape analysis
"""

from copy import deepcopy
from typing import Optional, Callable, List
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
from model.base_model import CL_Base_Model
import numpy as np
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds
from evaluations.expert_metrics import (
    LinearCKA, 
    HessianSpectrum, 
    EpsilonSharpness, 
    HessianDiagonal,
    LandscapeFlatness,
    MetricTimer,
    create_expert_loss_fn
)
from transformers import GenerationConfig
import json
import os
import math
import types
import time

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
        self.classifier = nn.Linear(config.hidden_size, num_experts)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        logits = self.classifier(hidden_states)
        routing_weights = F.softmax(logits, dim=1, dtype=torch.float)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, selected_experts


def convert_upcycle_model(model, args, num_tasks=0, incremental=False):
    """
    Convert the model to an Upcycling MoE model for inference.

    - If incremental == False: build model with num_tasks * num_experts_per_task experts.
    - If incremental == True: append num_tasks * num_experts_per_task new experts to existing MoE.
    """
    num_experts = int(getattr(args, 'num_experts_per_task', 8))
    device = torch.device("cuda")
    model_dtype = next(model.parameters()).dtype

    if incremental:
        num_new_experts = int(num_tasks) * num_experts
        if num_new_experts <= 0:
            print_rank_0(f"[Upcycle] incremental=True but num_new_experts=={num_new_experts}, nothing to do.", getattr(args, 'local_rank', 0))
            return model
        print_rank_0(f"⚙️ Incrementally adding {num_new_experts} experts per layer.", getattr(args, 'local_rank', 0))

        for li, layer in enumerate(model.model.layers):
            if li % 2 != 0:
                pass

            mlp = layer.mlp
            if not hasattr(mlp, "scientific_experts") or len(mlp.scientific_experts) == 0:
                mlp.original_forward = mlp.forward
                mlp.scientific_experts = nn.ModuleList([])
                if not hasattr(model.model.config, 'num_activated_experts'):
                    model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))
                mlp.router = Router(model.model.config, 0).to(device=device, dtype=model_dtype)

            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data

            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            new_intermediate_size = H // num_experts

            old_total = len(mlp.scientific_experts)
            for idx in range(num_new_experts):
                new_expert = Expert(model.model.config, new_intermediate_size).to(device=device, dtype=model_dtype)
                expert_idx_in_task = (old_total + idx) % num_experts
                start_col = expert_idx_in_task * new_intermediate_size
                end_col = (expert_idx_in_task + 1) * new_intermediate_size

                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

                mlp.scientific_experts.append(new_expert)

            old_classifier = mlp.router.classifier
            old_out = old_classifier.weight.data.shape[0] if hasattr(old_classifier, 'weight') and old_classifier.weight is not None else 0
            old_in = h
            new_out = old_out + num_new_experts
            new_router = nn.Linear(old_in, new_out).to(device=device, dtype=model_dtype)
            if old_out > 0:
                new_router.weight.data[:old_out, :].copy_(old_classifier.weight.data)
                if old_classifier.bias is not None:
                    new_router.bias.data[:old_out].copy_(old_classifier.bias.data)
            mlp.router.classifier = new_router

        return model

    # non-incremental: build total experts = num_tasks * num_experts
    num_total_experts = int(num_tasks) * num_experts
    print_rank_0(f"⚙️ Converting to Upcycling model with {num_total_experts} total experts for {num_tasks} tasks.", getattr(args, 'local_rank', 0))

    for li, layer in enumerate(model.model.layers):
        if li % 2 != 0:
            pass
        print_rank_0(f" Converting layer {li} to MoE layer.", getattr(args, 'local_rank', 0))

        mlp = layer.mlp
        mlp.original_forward = mlp.forward
        mlp.scientific_experts = nn.ModuleList([])

        if not hasattr(model.model.config, 'num_activated_experts'):
            model.model.config.num_activated_experts = int(getattr(args, 'num_activated_experts', 2))

        mlp.router = Router(model.model.config, num_total_experts).to(device=device, dtype=model_dtype)

        h = mlp.gate_proj.in_features
        H = mlp.gate_proj.out_features
        W_g = mlp.gate_proj.weight.data
        W_u = mlp.up_proj.weight.data
        W_d = mlp.down_proj.weight.data
        new_intermediate_size = H // num_experts

        for ei in range(num_total_experts):
            new_expert = Expert(model.model.config, new_intermediate_size).to(device=device, dtype=model_dtype)

            expert_idx_in_task = ei % num_experts
            start_col = expert_idx_in_task * new_intermediate_size
            end_col = (expert_idx_in_task + 1) * new_intermediate_size

            new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
            new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
            new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()

            mlp.scientific_experts.append(new_expert)

        layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)

    return model


def moe_forward(self, x):
    """
    Optimized MoE forward pass using batched expert computation.
    This function will be monkey-patched into each LlamaMLP instance.
    """
    # 1. Shared Expert (Original FFN)
    shared_expert_output = self.original_forward(x)

    # 2. Scientific Experts (batched implementation)
    if len(self.scientific_experts) > 0:
        # Handle both 2D (num_tokens, hidden_dim) and 3D (batch_size, seq_len, hidden_dim) inputs
        input_was_2d = x.dim() == 2
        if input_was_2d:
            # 2D input: (num_tokens, hidden_dim) -> add batch dimension
            num_tokens, hidden_dim = x.shape
            batch_size, seq_len = 1, num_tokens
            x_3d = x.unsqueeze(0)  # (1, num_tokens, hidden_dim)
        else:
            batch_size, seq_len, hidden_dim = x.shape
            num_tokens = batch_size * seq_len
            x_3d = x
        
        routing_weights, selected_experts = self.router(x_3d)
        # routing_weights: [num_tokens, top_k], selected_experts: [num_tokens, top_k]
        
        flat_x = x_3d.view(num_tokens, hidden_dim)
        
        # Initialize output
        final_expert_output = torch.zeros(num_tokens, hidden_dim, device=x.device, dtype=x.dtype)
        
        # Batch process by expert to avoid Python loops over tokens
        num_experts = len(self.scientific_experts)
        for expert_idx in range(num_experts):
            # Find all (token, slot) pairs routed to this expert
            # selected_experts: [num_tokens, top_k]
            expert_mask = (selected_experts == expert_idx)  # [num_tokens, top_k]
            
            if not expert_mask.any():
                continue
            
            # Get token indices that use this expert (in any slot)
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Get the tokens for this expert
            expert_input = flat_x[token_indices]  # [num_selected, hidden_dim]
            
            # Compute expert output
            expert_output = self.scientific_experts[expert_idx](expert_input)  # [num_selected, hidden_dim]
            
            # Get weights for these tokens (sum across slots where this expert is selected)
            token_weights = (routing_weights[token_indices] * expert_mask[token_indices].float()).sum(dim=-1, keepdim=True)
            
            # Accumulate weighted output
            final_expert_output[token_indices] += token_weights * expert_output
        
        # Reshape output: if input was 2D, remove batch dimension
        if input_was_2d:
            # This was originally 2D input, remove the added batch dimension
            final_expert_output = final_expert_output.view(num_tokens, hidden_dim)
            # shared_expert_output also needs to be 2D
            if shared_expert_output.dim() == 3:
                shared_expert_output = shared_expert_output.squeeze(0)
        else:
            final_expert_output = final_expert_output.view(batch_size, seq_len, hidden_dim)
        
        return shared_expert_output + final_expert_output
    else:
        return shared_expert_output


class Upcycle(CL_Base_Model):
    """
    Upcycling MoE Continual Learning Method.
    Converts the model to MoE architecture and adds new experts for each task.
    
    Includes comprehensive metric evaluation for expanded experts:
    - CKA (Centered Kernel Alignment) for representational similarity
    - Flatness/Sharpness metrics for loss landscape analysis
    """
    
    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        
        self.current_task_id = 0
        self.task2expert_range = {}
        
        # MoE configuration
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
        
        # Router initialization configuration
        # Options: 'random', 'average', 'zero_bias', 'scaled_random'
        self.router_init_method = getattr(args, 'router_init_method', 'random')

        # ==========================================================================
        # Metric Evaluation Configuration
        # ==========================================================================
        # Enable/disable metric evaluation
        self.enable_metrics = getattr(args, 'enable_expert_metrics', False)
        
        # CKA Configuration
        self.cka_mode = getattr(args, 'cka_mode', 'full')  # 'early' or 'full'
        self.cka_early_batches = getattr(args, 'cka_early_batches', 10)
        
        # Flatness/Sharpness Configuration
        self.sharpness_epsilon = getattr(args, 'sharpness_epsilon', 0.001)
        self.power_iterations = getattr(args, 'power_iterations', 20)
        self.hutchinson_samples = getattr(args, 'hutchinson_samples', 10)
        self.metric_num_batches = getattr(args, 'metric_num_batches', 5)
        
        # Flatness loss type: 'final' (default), 'reconstruction', or 'expert'
        # Default to 'final' for real loss computation
        # Use 'reconstruction' if OOM issues occur with final loss
        self.flatness_loss_type = getattr(args, 'flatness_loss_type', 'final')
        
        # Flatness computation batch size and sample limits
        # Adjusted based on expert size and available memory
        # For final loss, use minimal batches to avoid OOM
        if getattr(args, 'flatness_loss_type', 'final') == 'final':
            # Final loss requires full model forward pass, use minimal batches
            self.flatness_batch_size = getattr(args, 'flatness_batch_size', 2)  # Minimal batch size for final loss
            self.flatness_min_samples = getattr(args, 'flatness_min_samples', 64)  # Minimal samples for final loss
            self.flatness_max_samples = getattr(args, 'flatness_max_samples', 128)  # Minimal max samples for final loss
            self.final_loss_max_batches = getattr(args, 'final_loss_max_batches', 1)  # Minimal batches for final loss
        else:
            # Reconstruction/expert loss can use larger batches
            self.flatness_batch_size = getattr(args, 'flatness_batch_size', 32)
            self.flatness_min_samples = getattr(args, 'flatness_min_samples', 256)
            self.flatness_max_samples = getattr(args, 'flatness_max_samples', 512)
            self.final_loss_max_batches = None  # Not used for non-final loss
        
        # Evaluation checkpoints: 'after_train', 'during_train', 'both'
        self.metric_checkpoint = getattr(args, 'metric_checkpoint', 'after_train')
        self.metric_eval_interval = getattr(args, 'metric_eval_interval', 100)  # steps for during_train
        
        # Which layer's experts to evaluate (default: first MoE layer)
        # Set to -1 to evaluate ALL MoE layers
        self.metric_layer_idx = getattr(args, 'metric_layer_idx', 0)
        self.metric_all_layers = getattr(args, 'metric_all_layers', False)
        
        # Routing-aware metrics: whether to filter tokens based on top-k routing
        # Options: 'all' (use all tokens), 'routed' (only tokens routed to each expert)
        self.metric_routing_mode = getattr(args, 'metric_routing_mode', 'routed')
        
        # Expert scope: which experts to evaluate
        # 'current_task': only experts from current task (default)
        # 'all': all experts across all tasks
        self.metric_expert_scope = getattr(args, 'metric_expert_scope', 'all')
        
        # CKA activation source: which part of expert to use for CKA
        # 'output': full expert output (default)
        # 'up_proj': up_proj output (before multiplication with gate)
        # 'gate_proj': gate_proj output (after activation)
        self.cka_activation_source = getattr(args, 'cka_activation_source', 'output')
        
        # Flatness method: 'hessian' (default) or 'landscape' (from llm-landscape)
        # Hessian: Uses power iteration for λ_max, Hutchinson for trace (faster, exact curvature)
        # Landscape: Uses random direction perturbation (memory efficient, intuitive visualization)
        self.flatness_method = getattr(args, 'flatness_method', 'hessian')
        self.landscape_steps = getattr(args, 'landscape_steps', 10)
        self.landscape_multiplier = getattr(args, 'landscape_multiplier', 0.1)
        self.landscape_num_directions = getattr(args, 'landscape_num_directions', 3)
        
        # When no routed tokens are available, use landscape method as fallback
        # Landscape method doesn't depend on specific tokens, making it suitable for this case
        self.use_landscape_fallback = getattr(args, 'use_landscape_fallback', True)  # Default: enabled
        
        # Storage for metric results
        self.metric_results = {}
        
        # Flatness monitoring
        try:
            from evaluations.flatness_monitor import FlatnessMonitor
            monitor_output_dir = getattr(args, 'metric_output_dir', None)
            if monitor_output_dir:
                from pathlib import Path
                monitor_output_dir = Path(monitor_output_dir) / 'flatness_monitor'
            self.flatness_monitor = FlatnessMonitor(output_dir=monitor_output_dir)
        except ImportError:
            self.flatness_monitor = None
            print_rank_0("[Warning] FlatnessMonitor not available, monitoring disabled", 0)
        
        # Multi-scenario flatness evaluation configuration
        # Enable comprehensive flatness evaluation across different scenarios
        self.enable_multi_scenario_flatness = getattr(args, 'enable_multi_scenario_flatness', False)
        # If enabled, evaluates:
        # 1. Current task experts on current task data (training sufficiency)
        # 2. Current task experts on next task data (reusability without expansion)
        # 3. Expanded experts on next task data (post-expansion readiness)
        
        # Store old experts for CKA comparison (snapshots before training)
        self.old_expert_snapshots = {}
        
        # Separate optimizer for dynamically added experts (DeepSpeed ZeRO doesn't track them)
        self.expert_optimizer = None
        
        print_rank_0(f"[MoE-Upcycle] Initializing with {self.num_experts_per_task} experts per task, "
                     f"{self.num_activated_experts} activated, interval={self.upcycle_interval}", 
                     self.args.global_rank)
        
        if self.enable_metrics:
            print_rank_0(f"[MoE-Upcycle] Expert Metrics ENABLED: CKA mode={self.cka_mode}, "
                        f"flatness={self.flatness_method}, checkpoint={self.metric_checkpoint}, "
                        f"routing_mode={self.metric_routing_mode}", 
                        self.args.global_rank)

        # Device setup
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda", self.args.local_rank)
        
        # Get model dtype for consistency
        self.model_dtype = next(self.model.parameters()).dtype

    def train_continual(self):
        """Main continual learning loop with upcycling."""
        for i_task, task in enumerate(self.train_task_list):
            self.current_task_id = i_task
            print_rank_0(f"[MoE-Upcycle] >>>>> Start task-{i_task}: {task}", self.args.global_rank)

            # Decide whether to upcycle for this task
            do_upcycle = False
            if self.upcycle_interval and (i_task % max(1, self.upcycle_interval) == 0):
                do_upcycle = True
            if task in self.upcycle_task_names:
                do_upcycle = True

            if do_upcycle:
                self.upcycle_one_task(task, i_task)
            else:
                print_rank_0(f"[MoE-Upcycle] Skipping upcycle for task-{i_task}: {task}", self.args.global_rank)

            # Train the model on the current task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Save the model state after training
            self.save_model(i_task)

    def upcycle_one_task(self, task, i_task):
        """
        Adds and initializes a new set of experts for the current task.
        Uses "Scientific Expert Split" methodology.
        """
        if i_task in self.task2expert_range:
            print_rank_0(f"Task {i_task} already has experts. Skipping upcycling.", self.args.global_rank)
            return

        print_rank_0(f"Upcycling model for new task: {task} (Task ID: {i_task})", self.args.global_rank)
        
        # Check existing experts
        if hasattr(self.model.model.layers[0].mlp, "scientific_experts") and \
           len(self.model.model.layers[0].mlp.scientific_experts) > 0:
            total_experts_before = len(self.model.model.layers[0].mlp.scientific_experts)
            print_rank_0(f"Current experts per layer before upcycling: {total_experts_before}", self.args.global_rank)
        else:
            total_experts_before = 0
            print_rank_0(f"No existing experts found. Initializing MoE structure.", self.args.global_rank)

            # Initialize MoE structure for all layers
            for i, layer in enumerate(self.model.model.layers):
                if i % 2 != 0:
                    pass  # Skip odd layers or process all - configurable
                
                mlp = layer.mlp
                mlp.original_forward = mlp.forward
                mlp.scientific_experts = nn.ModuleList([])
                self.model.model.config.num_activated_experts = self.num_activated_experts
                mlp.router = Router(self.model.model.config, 0).to(device=self.device, dtype=self.model_dtype)
                layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
                print_rank_0(f"Initialized MoE structure for layer {i}", self.args.global_rank)

        # Record expert range for this task
        start_expert_idx = total_experts_before
        end_expert_idx = start_expert_idx + self.num_experts_per_task
        self.task2expert_range[i_task] = range(start_expert_idx, end_expert_idx)

        # Add new experts to each layer
        for i, layer in enumerate(self.model.model.layers):
            if i % 2 != 0:
                pass  # Skip odd layers or process all
            
            mlp = layer.mlp
            
            h = mlp.gate_proj.in_features
            H = mlp.gate_proj.out_features
            
            W_g = mlp.gate_proj.weight.data
            W_u = mlp.up_proj.weight.data
            W_d = mlp.down_proj.weight.data
            
            new_intermediate_size = H // self.num_experts_per_task
            print_rank_0(f"Layer {i}: intermediate size for new experts: {new_intermediate_size}", self.args.global_rank)

            # Create and initialize new experts
            for ei in range(self.num_experts_per_task):
                new_expert = Expert(self.model.model.config, new_intermediate_size).to(device=self.device, dtype=self.model_dtype)
                
                start_col = ei * new_intermediate_size
                end_col = (ei + 1) * new_intermediate_size
                
                # Copy sliced weights from shared FFN
                new_expert.gate_proj.weight.data = W_g[start_col:end_col, :].clone()
                new_expert.up_proj.weight.data = W_u[start_col:end_col, :].clone()
                new_expert.down_proj.weight.data = W_d[:, start_col:end_col].clone()
                
                mlp.scientific_experts.append(new_expert)
            
            # Expand router classifier
            num_total_experts = len(mlp.scientific_experts)
            new_router_classifier = nn.Linear(h, num_total_experts, device=self.device, dtype=self.model_dtype)
            if total_experts_before > 0:
                new_router_classifier.weight.data[:total_experts_before, :] = mlp.router.classifier.weight.data
                if mlp.router.classifier.bias is not None:
                    new_router_classifier.bias.data[:total_experts_before] = mlp.router.classifier.bias.data
            
            # Initialize new expert rows based on router_init_method
            self._init_router_new_experts(
                new_router_classifier, 
                total_experts_before, 
                num_total_experts,
                mlp.router.classifier if total_experts_before > 0 else None
            )
            
            mlp.router.classifier = new_router_classifier

        print_rank_0(f"Added {self.num_experts_per_task} new experts. "
                     f"Total experts per layer: {len(self.model.model.layers[0].mlp.scientific_experts)}", 
                     self.args.global_rank)
        
        # Create a separate optimizer for new expert parameters
        # DeepSpeed ZeRO doesn't track dynamically added parameters
        self._create_expert_optimizer(i_task)

    def _init_router_new_experts(
        self, 
        new_classifier: nn.Linear, 
        old_num_experts: int, 
        new_num_experts: int,
        old_classifier: nn.Linear = None
    ):
        """
        Initialize the router weights for newly added experts.
        
        Args:
            new_classifier: The new (expanded) router classifier
            old_num_experts: Number of experts before expansion
            new_num_experts: Total number of experts after expansion
            old_classifier: The old router classifier (for computing statistics)
        
        Initialization methods:
            - 'random': PyTorch default (Kaiming uniform) - may cause routing instability
            - 'average': Use average of old expert weights - smooth transition
            - 'zero_bias': Random weights but with negative bias - reduce initial routing to new experts
            - 'scaled_random': Smaller random weights - gradual integration
        """
        num_new = new_num_experts - old_num_experts
        if num_new <= 0:
            return
        
        method = self.router_init_method
        
        if method == 'random':
            # Keep PyTorch default initialization (already done)
            print_rank_0(f"   [Router Init] Using 'random' initialization for {num_new} new experts", 
                        self.args.global_rank)
            
        elif method == 'average':
            # Use average of old expert weights for smooth transition
            if old_classifier is not None and old_num_experts > 0:
                avg_weight = old_classifier.weight.data.mean(dim=0, keepdim=True)
                new_classifier.weight.data[old_num_experts:, :] = avg_weight.expand(num_new, -1)
                if new_classifier.bias is not None:
                    avg_bias = old_classifier.bias.data.mean() if old_classifier.bias is not None else 0.0
                    new_classifier.bias.data[old_num_experts:] = avg_bias
                print_rank_0(f"   [Router Init] Using 'average' initialization for {num_new} new experts", 
                            self.args.global_rank)
            else:
                print_rank_0(f"   [Router Init] No old experts, falling back to random", 
                            self.args.global_rank)
                
        elif method == 'zero_bias':
            # Random weights but with negative bias to reduce initial routing
            # This helps existing experts maintain their routing patterns initially
            if new_classifier.bias is not None:
                # Set negative bias so new experts get lower softmax scores initially
                new_classifier.bias.data[old_num_experts:] = -2.0  # ~exp(-2) ≈ 0.135 relative weight
            print_rank_0(f"   [Router Init] Using 'zero_bias' initialization (bias=-2.0) for {num_new} new experts", 
                        self.args.global_rank)
                
        elif method == 'scaled_random':
            # Smaller random weights for gradual integration
            scale = 0.1
            with torch.no_grad():
                new_classifier.weight.data[old_num_experts:, :] *= scale
                if new_classifier.bias is not None:
                    new_classifier.bias.data[old_num_experts:] *= scale
            print_rank_0(f"   [Router Init] Using 'scaled_random' (scale={scale}) for {num_new} new experts", 
                        self.args.global_rank)
                        
        elif method == 'copy_with_noise':
            # Copy average of old weights and add small noise for diversity
            if old_classifier is not None and old_num_experts > 0:
                avg_weight = old_classifier.weight.data.mean(dim=0, keepdim=True)
                noise_scale = 0.01
                noise = torch.randn(num_new, avg_weight.shape[1], 
                                   device=avg_weight.device, dtype=avg_weight.dtype) * noise_scale
                new_classifier.weight.data[old_num_experts:, :] = avg_weight.expand(num_new, -1) + noise
                if new_classifier.bias is not None and old_classifier.bias is not None:
                    avg_bias = old_classifier.bias.data.mean()
                    new_classifier.bias.data[old_num_experts:] = avg_bias
                print_rank_0(f"   [Router Init] Using 'copy_with_noise' (noise_scale={noise_scale}) for {num_new} new experts", 
                            self.args.global_rank)
            else:
                print_rank_0(f"   [Router Init] No old experts, falling back to random", 
                            self.args.global_rank)
        else:
            print_rank_0(f"   [Router Init] Unknown method '{method}', using random", 
                        self.args.global_rank)

    def _create_expert_optimizer(self, i_task):
        """
        Create a separate optimizer for newly added expert parameters.
        
        DeepSpeed ZeRO only manages parameters present during initialization.
        New experts need their own optimizer for gradient updates.
        """
        if i_task not in self.task2expert_range:
            return
        
        expert_range = self.task2expert_range[i_task]
        new_params = []
        
        for layer in self.model.model.layers:
            mlp = layer.mlp
            if hasattr(mlp, 'scientific_experts'):
                for expert_idx in expert_range:
                    if expert_idx < len(mlp.scientific_experts):
                        for param in mlp.scientific_experts[expert_idx].parameters():
                            new_params.append(param)
                
                # Also include router parameters
                if hasattr(mlp, 'router') and hasattr(mlp.router, 'classifier'):
                    for param in mlp.router.classifier.parameters():
                        new_params.append(param)
        
        if not new_params:
            print_rank_0(f"[Warning] No new parameters found for task {i_task}", self.args.global_rank)
            self.expert_optimizer = None
            return
        
        # Get learning rate from DeepSpeed config or use default
        lr = self.args.learning_rate if hasattr(self.args, 'learning_rate') else 1e-5
        
        # Create AdamW optimizer for new parameters
        self.expert_optimizer = torch.optim.AdamW(
            new_params, 
            lr=float(lr),
            weight_decay=0.0,
            betas=(0.9, 0.95)
        )
        
        print_rank_0(f"[Expert Optimizer] Created optimizer for {len(new_params)} new parameters (lr={lr})", 
                    self.args.global_rank)

    def freeze_non_current_task_params(self, i_task):
        """
        Freezes parameters except for current task's experts and router.
        """
        for layer in self.model.model.layers:
            mlp = layer.mlp
            # Freeze all experts and router first
            for expert in getattr(mlp, "scientific_experts", []):
                for param in expert.parameters():
                    param.requires_grad = False
            if hasattr(mlp, "router"):
                for param in mlp.router.parameters():
                    param.requires_grad = False
            # Freeze MLP backbone
            for name, param in mlp.named_parameters():
                if not name.startswith("scientific_experts") and not name.startswith("router"):
                    param.requires_grad = False

        # Unfreeze current task's experts and router
        expert_range_to_train = self.task2expert_range.get(i_task)
        if expert_range_to_train is None:
            return

        print_rank_0(f"Unfreezing experts in range {expert_range_to_train} for task {i_task}", self.args.global_rank)
        
        for i, layer in enumerate(self.model.model.layers):
            if i % 2 != 0:
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
        """Train on a single task with frozen non-current parameters."""
        dataloader_train = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]
        total_steps = epochs * len(dataloader_train)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))

        # Freeze all params except those for the current task
        self.freeze_non_current_task_params(i_task)
        
        # Snapshot old experts before training (for CKA comparison)
        if self.enable_metrics and i_task in self.task2expert_range:
            self._snapshot_old_experts(i_task)

        global_step = 0
        self.model.train()
        for epoch in range(epochs):
            for step, batch in enumerate(dataloader_train):
                del batch['sources']
                batch = to_device(batch, self.device)
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Task-{i_task} Epoch-{epoch} Step-{step} loss={loss.item():.4f}")
                
                self.model.backward(loss)
                self.model.step()
                
                # Step the expert optimizer for dynamically added parameters
                if hasattr(self, 'expert_optimizer') and self.expert_optimizer is not None:
                    self.expert_optimizer.step()
                    self.expert_optimizer.zero_grad()
                
                global_step += 1
                
                # During-training metric evaluation
                if self.enable_metrics and self.metric_checkpoint in ['during_train', 'both']:
                    if global_step % self.metric_eval_interval == 0:
                        print_rank_0(f"\n[Metrics] Evaluating at step {global_step}...", self.args.global_rank)
                        self._evaluate_experts_metrics(
                            i_task, task, eval_dataloader, 
                            checkpoint_name=f"step_{global_step}"
                        )
                        self.model.train()  # Resume training mode
        
        # After-training metric evaluation
        if self.enable_metrics and self.metric_checkpoint in ['after_train', 'both']:
            print_rank_0(f"\n[Metrics] Evaluating after training task {i_task}...", self.args.global_rank)
            
            # Diagnostic: Check if expert weights actually changed during training
            if i_task in self.old_expert_snapshots and i_task in self.task2expert_range:
                expert_range = self.task2expert_range[i_task]
                layer_idx = self.metric_layer_idx
                for i, layer in enumerate(self.model.model.layers):
                    if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                        if layer_idx == 0:
                            layer_idx = i
                            break
                        layer_idx -= 1
                mlp = self.model.model.layers[layer_idx].mlp
                
                print_rank_0(f"[Metrics] Post-training weight check:", self.args.global_rank)
                for expert_idx in expert_range:
                    if expert_idx in self.old_expert_snapshots[i_task]:
                        old_expert = self.old_expert_snapshots[i_task][expert_idx]
                        new_expert = mlp.scientific_experts[expert_idx]
                        weight_diff = self._compute_weight_difference(old_expert, new_expert)
                        with torch.no_grad():
                            new_norm = sum(p.data.norm().item() for p in new_expert.parameters())
                        print_rank_0(f"   Expert {expert_idx}: weight_diff={weight_diff:.6e}, new_norm={new_norm:.6f}", 
                                   self.args.global_rank)
            
            self._evaluate_experts_metrics(
                i_task, task, eval_dataloader, 
                checkpoint_name="after_train"
            )
    
    # ==========================================================================
    # Expert Metrics Evaluation Methods
    # ==========================================================================
    
    def _clear_cuda_cache(self):
        """Clear CUDA cache and garbage collect to free memory."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def _cache_dataloader_batches(
        self,
        dataloader,
        num_batches: int = 20,
        min_batches: int = 20
    ) -> list:
        """
        Cache batches from dataloader for metric evaluation.
        
        Args:
            dataloader: DataLoader to cache batches from
            num_batches: Number of batches to cache (can be overridden by min_batches)
            min_batches: Minimum number of batches to cache
            
        Returns:
            List of cached batches (on CPU)
        """
        cached_batches = []
        num_batches_to_cache = max(num_batches, min_batches)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches_to_cache:
                break
            # Remove non-essential fields and move to CPU for caching
            if 'sources' in batch:
                del batch['sources']
            cached_batch = {
                'input_ids': batch['input_ids'].cpu(),
                'attention_mask': batch.get('attention_mask')
            }
            if cached_batch['attention_mask'] is not None:
                cached_batch['attention_mask'] = cached_batch['attention_mask'].cpu()
            cached_batches.append(cached_batch)
        
        return cached_batches
    
    def _get_routed_tokens_if_needed(
        self,
        hidden_states: torch.Tensor,
        mlp,
        expert_range: range,
        routing_mode: str,
        min_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """
        Get routed tokens for experts if routing mode is enabled.
        
        Args:
            hidden_states: Hidden states tensor
            mlp: MLP module with router
            expert_range: Range of expert indices
            routing_mode: 'routed' or 'all'
            min_tokens: Minimum tokens per expert (defaults to flatness_min_samples // 8)
            
        Returns:
            Dictionary mapping expert_idx to (hidden_states, routing_stats) or None
        """
        if routing_mode == 'routed' and hasattr(mlp, 'router'):
            if min_tokens is None:
                min_tokens = max(32, self.flatness_min_samples // 8)
            return self._get_all_experts_routed_tokens(
                hidden_states, mlp.router, expert_range, min_tokens=min_tokens
            )
        return None
    
    def _get_moe_layers(self) -> list:
        """Get all MoE layer indices."""
        moe_layers = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                moe_layers.append(i)
        return moe_layers
    
    def _get_experts_to_evaluate(self, i_task: int, mlp) -> range:
        """
        Get the range of experts to evaluate based on metric_expert_scope.
        
        Args:
            i_task: Current task ID
            mlp: The MLP module containing experts
            
        Returns:
            Range of expert indices to evaluate
        """
        if self.metric_expert_scope == 'all':
            # Evaluate all experts
            return range(len(mlp.scientific_experts))
        else:
            # Only current task's experts
            return self.task2expert_range.get(i_task, range(0))
    
    def _snapshot_old_experts(self, i_task: int):
        """
        Create a snapshot (deep copy) of experts before training for CKA comparison.
        
        Snapshots ALL experts (not just current task's) if metric_expert_scope='all'.
        Snapshots experts from all MoE layers if metric_all_layers=True.
        
        Args:
            i_task: Current task ID
        """
        if i_task not in self.task2expert_range:
            return
        
        self.old_expert_snapshots[i_task] = {}
        
        # Determine which layers to snapshot
        if self.metric_all_layers:
            layer_indices = self._get_moe_layers()
        else:
            # Find the specified MoE layer
            layer_indices = []
            layer_idx = self.metric_layer_idx
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                    if layer_idx == 0:
                        layer_indices = [i]
                        break
                    layer_idx -= 1
        
        print_rank_0(f"[Metrics] Snapshotting experts from {len(layer_indices)} layer(s): {layer_indices}", 
                    self.args.global_rank)
        
        for layer_idx in layer_indices:
            mlp = self.model.model.layers[layer_idx].mlp
            
            # Determine which experts to snapshot
            expert_range = self._get_experts_to_evaluate(i_task, mlp)
            
            for expert_idx in expert_range:
                if expert_idx < len(mlp.scientific_experts):
                    expert = mlp.scientific_experts[expert_idx]
                    # Deep copy the expert to CPU to save GPU memory
                    snapshot = deepcopy(expert).cpu()
                    
                    # Key format: layer_idx_expert_idx
                    key = f"L{layer_idx}_E{expert_idx}"
                    self.old_expert_snapshots[i_task][key] = snapshot
                    
                    # Diagnostic: compute weight norm to verify snapshot
                    with torch.no_grad():
                        orig_norm = sum(p.data.norm().item() for p in expert.parameters())
                        snap_norm = sum(p.data.norm().item() for p in snapshot.parameters())
                    print_rank_0(f"   {key}: orig_norm={orig_norm:.6f}, snap_norm={snap_norm:.6f}", 
                                self.args.global_rank)
        
        print_rank_0(f"[Metrics] Snapshot created for {len(self.old_expert_snapshots[i_task])} experts "
                    f"in task {i_task} (BEFORE training, scope={self.metric_expert_scope})", 
                    self.args.global_rank)
    
    def store_expansion_baseline(self, dataloader, num_batches: int = 10):
        """
        Store layer outputs BEFORE expansion for later CKA comparison.
        
        Call this method BEFORE calling _add_new_experts() to capture the
        pre-expansion layer outputs. After expansion, the metrics evaluation
        will compare the new outputs to these baselines.
        
        This enables measuring:
        - How much does layer output change due to adding new experts?
        - Is the representation stable after expansion?
        
        Args:
            dataloader: DataLoader for input samples
            num_batches: Number of batches to use for baseline computation
        """
        print_rank_0(f"[Metrics] Storing expansion baseline (pre-expansion layer outputs)...", 
                    self.args.global_rank)
        
        self._expansion_baseline = {}
        layer_indices = self._get_moe_layers()
        
        # Collect input samples
        all_hidden_states = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Get hidden states for each layer using model's forward with output_hidden_states
                if hasattr(self.model, 'module'):
                    base_model = self.model.module
                else:
                    base_model = self.model
                
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden_states_all = outputs.hidden_states
                
                # Store hidden states for each layer (input to that layer)
                for layer_idx in layer_indices:
                    if layer_idx not in all_hidden_states:
                        all_hidden_states[layer_idx] = []
                    # hidden_states_all[layer_idx] is the input to layer layer_idx
                    hs = hidden_states_all[layer_idx].view(-1, hidden_states_all[layer_idx].size(-1))
                    all_hidden_states[layer_idx].append(hs.cpu())
                
                del outputs, hidden_states_all
        
        # Compute layer outputs and store as baseline
        for layer_idx in layer_indices:
            if layer_idx not in all_hidden_states:
                continue
            
            # Concatenate all hidden states for this layer
            hidden_states = torch.cat(all_hidden_states[layer_idx], dim=0).to(self.device)
            
            # Limit samples for memory efficiency
            if hidden_states.size(0) > 512:
                hidden_states = hidden_states[:512]
            
            # Get layer MLP and compute output
            mlp = self.model.model.layers[layer_idx].mlp
            with torch.no_grad():
                output = mlp(hidden_states)
                if isinstance(output, tuple):
                    output = output[0]
                # Store on CPU to save GPU memory
                self._expansion_baseline[layer_idx] = output.cpu()
            
            del hidden_states, output
        
        self._clear_cuda_cache()
        print_rank_0(f"[Metrics] Expansion baseline stored for {len(self._expansion_baseline)} layers", 
                    self.args.global_rank)
    
    def clear_expansion_baseline(self):
        """Clear the expansion baseline to free memory."""
        if hasattr(self, '_expansion_baseline'):
            del self._expansion_baseline
            self._expansion_baseline = None
        self._clear_cuda_cache()
    
    def _get_expert_extractor(self, activation_source: str = 'output'):
        """
        Create activation extraction function for experts.
        
        Args:
            activation_source: Which activation to extract
                - 'output': Full expert output (default)
                - 'up_proj': up_proj output (captures input transformation)
                - 'gate_proj': gate_proj output after activation
                - 'gate_up': Concatenation of gate and up (richer representation)
        
        Returns:
            Callable that extracts activations from an expert
        """
        def extract_output(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Extract full expert output."""
            with torch.no_grad():
                output = expert(x)
                if isinstance(output, tuple):
                    output = output[0]
                return output
        
        def extract_up_proj(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Extract up_proj output (linear transformation without gating)."""
            with torch.no_grad():
                return expert.up_proj(x)
        
        def extract_gate_proj(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Extract gate_proj output after activation."""
            with torch.no_grad():
                return expert.act_fn(expert.gate_proj(x))
        
        def extract_gate_up(expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Extract concatenation of gate and up (richer representation)."""
            with torch.no_grad():
                gate_out = expert.act_fn(expert.gate_proj(x))
                up_out = expert.up_proj(x)
                return torch.cat([gate_out, up_out], dim=-1)
        
        extractors = {
            'output': extract_output,
            'up_proj': extract_up_proj,
            'gate_proj': extract_gate_proj,
            'gate_up': extract_gate_up
        }
        
        return extractors.get(activation_source, extract_output)
    
    def _compute_weight_difference(self, expert1: nn.Module, expert2: nn.Module) -> float:
        """
        Compute L2 distance between two experts' weights.
        
        This helps diagnose if the snapshot was correctly captured.
        A value of 0.0 means weights are identical (no training happened or snapshot issue).
        
        Args:
            expert1: First expert (typically old/snapshot)
            expert2: Second expert (typically new/current)
            
        Returns:
            L2 distance of flattened weight vectors
        """
        with torch.no_grad():
            params1 = []
            params2 = []
            
            for p in expert1.parameters():
                params1.append(p.detach().cpu().float().flatten())
            for p in expert2.parameters():
                params2.append(p.detach().cpu().float().flatten())
            
            if not params1 or not params2:
                return 0.0
            
            vec1 = torch.cat(params1)
            vec2 = torch.cat(params2)
            
            if vec1.shape != vec2.shape:
                return -1.0  # Shape mismatch
            
            diff = torch.norm(vec1 - vec2).item()
            return diff
    
    def _prepare_inputs_for_metrics_from_cache(
        self, 
        cached_batches: list, 
        max_samples: int = 512,
        target_layer_idx: int = 0
    ):
        """
        Prepare input tensors for metric evaluation from pre-cached batches.
        
        This avoids dataloader exhaustion issues when evaluating multiple layers.
        Unified method that can also accept a dataloader (for backward compatibility).
        
        Args:
            cached_batches: List of cached batch dicts with 'input_ids' and 'attention_mask',
                          or a dataloader (will be converted to list)
            max_samples: Maximum number of samples to collect
            target_layer_idx: Which layer to prepare inputs for. If None, uses the first MoE layer.
            
        Returns:
            Tensor of hidden states (num_samples, hidden_dim)
        """
        # Handle dataloader input (for backward compatibility)
        if not isinstance(cached_batches, list):
            # Convert dataloader to list of batches
            dataloader = cached_batches
            cached_batches = []
            num_batches = getattr(self, 'metric_num_batches', 5)
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_batches:
                    break
                if 'sources' in batch:
                    del batch['sources']
                cached_batch = {
                    'input_ids': batch['input_ids'].cpu(),
                    'attention_mask': batch.get('attention_mask')
                }
                if cached_batch['attention_mask'] is not None:
                    cached_batch['attention_mask'] = cached_batch['attention_mask'].cpu()
                cached_batches.append(cached_batch)
        
        # Find the target MoE layer index if not specified
        if target_layer_idx is None:
            target_layer_idx = self.metric_layer_idx
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                    if target_layer_idx == 0:
                        target_layer_idx = i
                        break
                    target_layer_idx -= 1
        self.model.eval()
        hidden_states_list = []
        total_samples = 0
        
        # Clear cache before starting
        self._clear_cuda_cache()
        
        try:
            with torch.no_grad():
                for batch in cached_batches:
                    if total_samples >= max_samples:
                        break
                    
                    # Move to GPU
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    
                    # Get the base model (the inner LlamaModel, not the LlamaForCausalLM wrapper)
                    if hasattr(self.model, 'module'):
                        base_model = self.model.module.model
                    else:
                        base_model = self.model.model
                    
                    # Use model's forward with output_hidden_states to get all layer outputs
                    # This properly handles position embeddings, attention masks, etc.
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    
                    # hidden_states is a tuple of (embedding_output, layer_0_output, layer_1_output, ...)
                    # For layer N, we want the input to that layer, which is the output of layer N-1
                    # So for target_layer_idx=0, we want embedding output (index 0)
                    # For target_layer_idx=1, we want layer 0 output (index 1)
                    all_hidden_states = outputs.hidden_states
                    hidden_states = all_hidden_states[target_layer_idx]  # Input to target layer
                    
                    # Flatten and collect (batch_size * seq_len, hidden_dim)
                    batch_hidden = hidden_states.view(-1, hidden_states.size(-1))
                    
                    # Limit samples
                    samples_to_take = min(batch_hidden.size(0), max_samples - total_samples)
                    hidden_states_list.append(batch_hidden[:samples_to_take].cpu())
                    total_samples += samples_to_take
                    
                    # Clear GPU memory
                    del hidden_states, input_ids, batch_hidden, outputs, all_hidden_states
                    if attention_mask is not None:
                        del attention_mask
                    self._clear_cuda_cache()
                    
        except Exception as e:
            print_rank_0(f"[Metrics] Warning: Error preparing inputs from cache: {e}", self.args.global_rank)
            import traceback
            traceback.print_exc()
            self._clear_cuda_cache()
            return None
        
        if hidden_states_list:
            all_hidden = torch.cat(hidden_states_list, dim=0)
            return all_hidden.to(self.device)
        return None
    
    def _evaluate_experts_metrics(
        self, 
        i_task: int, 
        task: str, 
        dataloader,
        checkpoint_name: str = "checkpoint"
    ):
        """
        Main entry point for expert metrics evaluation.
        Coordinates all evaluation modules: diversity, CKA, and flatness.
        
        Args:
            i_task: Task ID
            task: Task name
            dataloader: Evaluation dataloader
            checkpoint_name: Name for this checkpoint (e.g., 'step_100', 'after_train')
        """
        if self.args.global_rank != 0:
            # Only evaluate on main process
            return
        
        total_start = time.perf_counter()
        
        # Clear CUDA cache before starting evaluation
        print_rank_0(f"\n[Metrics] Clearing CUDA cache before evaluation...", 0)
        self._clear_cuda_cache()
        
        print_rank_0(f"\n{'='*60}", 0)
        print_rank_0(f"Expert Metrics Evaluation - Task {i_task}: {task}", 0)
        print_rank_0(f"Checkpoint: {checkpoint_name}", 0)
        print_rank_0(f"Scope: {self.metric_expert_scope}, All Layers: {self.metric_all_layers}", 0)
        print_rank_0(f"CKA Source: {self.cka_activation_source}, Flatness Method: {self.flatness_method}", 0)
        print_rank_0(f"{'='*60}", 0)
        
        # Detect if this is initial flatness (after expansion) or trained flatness
        is_initial_flatness = hasattr(self, '_expansion_baseline') and self._expansion_baseline is not None
        flatness_type = 'initial' if is_initial_flatness else 'trained'
        
        results = {
            'task_id': i_task,
            'task_name': task,
            'checkpoint': checkpoint_name,
            'flatness_type': flatness_type,
            'config': {
                'metric_expert_scope': self.metric_expert_scope,
                'metric_all_layers': self.metric_all_layers,
                'cka_activation_source': self.cka_activation_source,
                'flatness_method': self.flatness_method,
                'flatness_loss_type': self.flatness_loss_type,
                'metric_routing_mode': self.metric_routing_mode
            },
            'cka': {},
            'flatness': {},
            'flatness_initial': {},
            'flatness_trained': {}
        }
        
        self.model.eval()
        
        # Determine which layers to evaluate
        layer_indices = self._get_layers_to_evaluate()
        if not layer_indices:
            print_rank_0("[Metrics] No MoE layers found", 0)
            return
        
        print_rank_0(f"\n[Metrics] Evaluating {len(layer_indices)} layer(s): {layer_indices}", 0)
        
        # Cache batches from dataloader
        num_batches_to_cache = max(
            self.cka_early_batches if self.cka_mode == 'early' else self.metric_num_batches,
            20
        )
        print_rank_0(f"[Metrics] Caching {num_batches_to_cache} batches from dataloader...", 0)
        cached_batches = self._cache_dataloader_batches(
            dataloader,
            num_batches=num_batches_to_cache,
            min_batches=20
        )
        print_rank_0(f"[Metrics] Cached {len(cached_batches)} batches", 0)
        
        try:
            # Evaluate different metrics using dedicated methods
            results.update(self._evaluate_expert_diversity(i_task, layer_indices, cached_batches))
            results.update(self._evaluate_layer_cka(i_task, layer_indices, cached_batches))
            results.update(self._evaluate_expert_cka(i_task, layer_indices, cached_batches))
            results.update(self._evaluate_expert_flatness(i_task, task, layer_indices, cached_batches))
        
        except Exception as e:
            print_rank_0(f"[Metrics] Error during evaluation: {e}", 0)
            import traceback
            traceback.print_exc()
        finally:
            # Final cleanup - release cached batches
            if 'cached_batches' in locals():
                del cached_batches
            self._clear_cuda_cache()
        
        return results
    
    def _get_layers_to_evaluate(self) -> list:
        """
        Determine which MoE layers to evaluate based on configuration.
        
        Returns:
            List of layer indices to evaluate
        """
        if self.metric_all_layers:
            return self._get_moe_layers()
        else:
            # Find the specified MoE layer
            layer_indices = []
            layer_idx = self.metric_layer_idx
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                    if layer_idx == 0:
                        layer_indices = [i]
                        break
                    layer_idx -= 1
            return layer_indices
    
    def _evaluate_expert_diversity(
        self, 
        i_task: int, 
        layer_indices: list, 
        cached_batches: list
    ) -> dict:
        """
        Evaluate inter-expert diversity using pairwise CKA.
        
        Returns:
            Dictionary with diversity results
        """
        results = {'diversity': {'global': {}, 'task': {}}}
        
        try:
            print_rank_0(f"\n📐 Expert Diversity Evaluation", 0)
            print_rank_0(f"   ┌─────────────────────────────────────────────────────────────────┐", 0)
            print_rank_0(f"   │ Diversity = 1 - mean(pairwise CKA between experts)              │", 0)
            print_rank_0(f"   │   • 0.0 = All experts produce identical outputs (redundant)    │", 0)
            print_rank_0(f"   │   • 1.0 = Experts are completely different (specialized)       │", 0)
            print_rank_0(f"   │ Global: Diversity among ALL experts (E0-En)                    │", 0)
            print_rank_0(f"   │ Task:   Diversity among current task's experts only            │", 0)
            print_rank_0(f"   └─────────────────────────────────────────────────────────────────┘", 0)
            
            from evaluations.representation_metrics import RepresentationAlignmentEvaluator
            rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
            
            current_task_expert_range = self.task2expert_range.get(i_task, range(0))
            
            for layer_idx in layer_indices:
                if hasattr(self.model, 'module'):
                    mlp = self.model.module.model.layers[layer_idx].mlp
                else:
                    mlp = self.model.model.layers[layer_idx].mlp
                
                if not hasattr(mlp, 'scientific_experts') or len(mlp.scientific_experts) == 0:
                    continue
                
                all_experts = list(mlp.scientific_experts)
                current_task_experts = [
                    mlp.scientific_experts[idx] 
                    for idx in current_task_expert_range 
                    if idx < len(mlp.scientific_experts)
                ]
                
                hidden_states = self._prepare_inputs_for_metrics(
                    cached_batches, max_samples=256, target_layer_idx=layer_idx
                )
                if hidden_states is None:
                    continue
                
                hidden_states = hidden_states.to(self.device)
                
                is_first_layer = (layer_idx == layer_indices[0])
                if is_first_layer:
                    print_rank_0(f"   [Debug] Hidden states shape: {hidden_states.shape}", 0)
                    print_rank_0(f"   [Debug] Hidden states stats: mean={hidden_states.mean():.4f}, "
                                f"std={hidden_states.std():.4f}", 0)
                
                # Global Diversity
                if len(all_experts) >= 2:
                    cka_matrix_global, diversity_global = rep_evaluator.compute_inter_expert_diversity(
                        all_experts, hidden_states, 
                        max_experts=len(all_experts),
                        activation_source='up_proj',
                        verbose=is_first_layer
                    )
                    results['diversity']['global'][f'layer_{layer_idx}'] = {
                        'diversity_score': diversity_global,
                        'num_experts': len(all_experts),
                        'mean_pairwise_cka': 1.0 - diversity_global if diversity_global >= 0 else -1.0
                    }
                else:
                    diversity_global = None
                
                # Task Diversity
                if len(current_task_experts) >= 2:
                    cka_matrix_task, diversity_task = rep_evaluator.compute_inter_expert_diversity(
                        current_task_experts, hidden_states, 
                        max_experts=len(current_task_experts),
                        activation_source='up_proj',
                        verbose=False
                    )
                    results['diversity']['task'][f'layer_{layer_idx}'] = {
                        'diversity_score': diversity_task,
                        'num_experts': len(current_task_experts),
                        'expert_range': list(current_task_expert_range),
                        'mean_pairwise_cka': 1.0 - diversity_task if diversity_task >= 0 else -1.0
                    }
                else:
                    diversity_task = None
                
                # Print results
                n_global = len(all_experts)
                n_task = len(current_task_experts)
                if diversity_global is not None and diversity_task is not None:
                    global_cka = 1.0 - diversity_global
                    task_cka = 1.0 - diversity_task
                    print_rank_0(f"   L{layer_idx:2d}: Global div={diversity_global:.4f} (CKA={global_cka:.4f}, {n_global}E) | "
                                f"Task{i_task} div={diversity_task:.4f} (CKA={task_cka:.4f}, E{list(current_task_expert_range)[0]}-{list(current_task_expert_range)[-1]})", 0)
                elif diversity_global is not None:
                    global_cka = 1.0 - diversity_global
                    print_rank_0(f"   L{layer_idx:2d}: Global div={diversity_global:.4f} (CKA={global_cka:.4f}, {n_global}E) | Task{i_task}=N/A", 0)
                else:
                    print_rank_0(f"   L{layer_idx:2d}: Global=N/A | Task{i_task}=N/A", 0)
                
                del hidden_states
                self._clear_cuda_cache()
                
        except Exception as e:
            print_rank_0(f"[Warning] Diversity evaluation failed: {e}", 0)
            import traceback
            traceback.print_exc()
        
        return results
        
        total_time = time.perf_counter() - total_start
        results['total_time'] = total_time
        
        # Print summary
        self._print_metrics_summary(results)
        
        print_rank_0(f"\n⏱️  Total Evaluation Time: {total_time:.4f}s", 0)
        print_rank_0(f"{'='*60}\n", 0)
        
        # Store results
        result_key = f"task_{i_task}_{checkpoint_name}"
        self.metric_results[result_key] = results
        
        # Update flatness monitor
        if self.flatness_monitor is not None:
            self.flatness_monitor.load_from_results(results)
        
        # Save results to file
        self._save_metric_results(i_task, checkpoint_name, results)
        
        # Update flatness monitor
        if self.flatness_monitor is not None:
            try:
                self.flatness_monitor.load_from_results(results)
            except Exception as e:
                print_rank_0(f"[Warning] Failed to update flatness monitor: {e}", 0)
    
    def _get_layers_to_evaluate(self) -> list:
        """
        Determine which MoE layers to evaluate based on configuration.
        
        Returns:
            List of layer indices to evaluate
        """
        if self.metric_all_layers:
            return self._get_moe_layers()
        else:
            # Find the specified MoE layer
            layer_indices = []
            layer_idx = self.metric_layer_idx
            for i, layer in enumerate(self.model.model.layers):
                if hasattr(layer.mlp, 'scientific_experts') and len(layer.mlp.scientific_experts) > 0:
                    if layer_idx == 0:
                        layer_indices = [i]
                        break
                    layer_idx -= 1
            return layer_indices
    
    def _evaluate_expert_diversity(
        self, 
        i_task: int, 
        layer_indices: list, 
        cached_batches: list
    ) -> dict:
        """
        Evaluate inter-expert diversity using pairwise CKA.
        
        Returns:
            Dictionary with diversity results
        """
        results = {'diversity': {'global': {}, 'task': {}}}
        
        try:
            print_rank_0(f"\n📐 Expert Diversity Evaluation", 0)
            print_rank_0(f"   ┌─────────────────────────────────────────────────────────────────┐", 0)
            print_rank_0(f"   │ Diversity = 1 - mean(pairwise CKA between experts)              │", 0)
            print_rank_0(f"   │   • 0.0 = All experts produce identical outputs (redundant)    │", 0)
            print_rank_0(f"   │   • 1.0 = Experts are completely different (specialized)       │", 0)
            print_rank_0(f"   │ Global: Diversity among ALL experts (E0-En)                    │", 0)
            print_rank_0(f"   │ Task:   Diversity among current task's experts only            │", 0)
            print_rank_0(f"   └─────────────────────────────────────────────────────────────────┘", 0)
            
            from evaluations.representation_metrics import RepresentationAlignmentEvaluator
            rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
            
            current_task_expert_range = self.task2expert_range.get(i_task, range(0))
            
            for layer_idx in layer_indices:
                if hasattr(self.model, 'module'):
                    mlp = self.model.module.model.layers[layer_idx].mlp
                else:
                    mlp = self.model.model.layers[layer_idx].mlp
                
                if not hasattr(mlp, 'scientific_experts') or len(mlp.scientific_experts) == 0:
                    continue
                
                all_experts = list(mlp.scientific_experts)
                current_task_experts = [
                    mlp.scientific_experts[idx] 
                    for idx in current_task_expert_range 
                    if idx < len(mlp.scientific_experts)
                ]
                
                hidden_states = self._prepare_inputs_for_metrics_from_cache(
                    cached_batches, max_samples=256, target_layer_idx=layer_idx
                )
                if hidden_states is None:
                    continue
                
                hidden_states = hidden_states.to(self.device)
                
                is_first_layer = (layer_idx == layer_indices[0])
                if is_first_layer:
                    print_rank_0(f"   [Debug] Hidden states shape: {hidden_states.shape}", 0)
                    print_rank_0(f"   [Debug] Hidden states stats: mean={hidden_states.mean():.4f}, "
                                f"std={hidden_states.std():.4f}", 0)
                
                # Global Diversity
                if len(all_experts) >= 2:
                    cka_matrix_global, diversity_global = rep_evaluator.compute_inter_expert_diversity(
                        all_experts, hidden_states, 
                        max_experts=len(all_experts),
                        activation_source='up_proj',
                        verbose=is_first_layer
                    )
                    results['diversity']['global'][f'layer_{layer_idx}'] = {
                        'diversity_score': diversity_global,
                        'num_experts': len(all_experts),
                        'mean_pairwise_cka': 1.0 - diversity_global if diversity_global >= 0 else -1.0
                    }
                else:
                    diversity_global = None
                
                # Task Diversity
                if len(current_task_experts) >= 2:
                    cka_matrix_task, diversity_task = rep_evaluator.compute_inter_expert_diversity(
                        current_task_experts, hidden_states, 
                        max_experts=len(current_task_experts),
                        activation_source='up_proj',
                        verbose=False
                    )
                    results['diversity']['task'][f'layer_{layer_idx}'] = {
                        'diversity_score': diversity_task,
                        'num_experts': len(current_task_experts),
                        'expert_range': list(current_task_expert_range),
                        'mean_pairwise_cka': 1.0 - diversity_task if diversity_task >= 0 else -1.0
                    }
                else:
                    diversity_task = None
                
                # Print results
                n_global = len(all_experts)
                n_task = len(current_task_experts)
                if diversity_global is not None and diversity_task is not None:
                    global_cka = 1.0 - diversity_global
                    task_cka = 1.0 - diversity_task
                    print_rank_0(f"   L{layer_idx:2d}: Global div={diversity_global:.4f} (CKA={global_cka:.4f}, {n_global}E) | "
                                f"Task{i_task} div={diversity_task:.4f} (CKA={task_cka:.4f}, E{list(current_task_expert_range)[0]}-{list(current_task_expert_range)[-1]})", 0)
                elif diversity_global is not None:
                    global_cka = 1.0 - diversity_global
                    print_rank_0(f"   L{layer_idx:2d}: Global div={diversity_global:.4f} (CKA={global_cka:.4f}, {n_global}E) | Task{i_task}=N/A", 0)
                else:
                    print_rank_0(f"   L{layer_idx:2d}: Global=N/A | Task{i_task}=N/A", 0)
                
                del hidden_states
                self._clear_cuda_cache()
                
        except Exception as e:
            print_rank_0(f"[Warning] Diversity evaluation failed: {e}", 0)
            import traceback
            traceback.print_exc()
        
        return results
    
    def _evaluate_layer_cka(
        self, 
        i_task: int, 
        layer_indices: list, 
        cached_batches: list
    ) -> dict:
        """
        Evaluate layer-level CKA (representation stability after expansion).
        
        Returns:
            Dictionary with layer CKA results
        """
        results = {'layer_cka': {}}
        
        if not (hasattr(self, '_expansion_baseline') and self._expansion_baseline is not None):
            print_rank_0(f"   [Info] No expansion baseline available for layer CKA comparison", 0)
            print_rank_0(f"   [Tip] To enable, call store_expansion_baseline() before expansion", 0)
            return results
        
        try:
            print_rank_0(f"\n🔄 Layer-Level CKA (Representation Stability)", 0)
            print_rank_0(f"   ┌─────────────────────────────────────────────────────────────────┐", 0)
            print_rank_0(f"   │ Measures how layer output changes due to expert expansion       │", 0)
            print_rank_0(f"   │ Token Usage:                                                    │", 0)
            print_rank_0(f"   │   • Layer CKA: ALL tokens (comprehensive layer comparison)     │", 0)
            print_rank_0(f"   │   • Expert CKA: ROUTED tokens (expert-specific comparison)     │", 0)
            print_rank_0(f"   │   • Flatness: ROUTED tokens (expert-specific loss landscape)   │", 0)
            print_rank_0(f"   └─────────────────────────────────────────────────────────────────┘", 0)
            
            from evaluations.representation_metrics import RepresentationAlignmentEvaluator
            rep_evaluator = RepresentationAlignmentEvaluator(device=self.device)
            
            for layer_idx in layer_indices:
                if layer_idx not in self._expansion_baseline:
                    continue
                
                hidden_states = self._prepare_inputs_for_metrics_from_cache(
                    cached_batches, max_samples=256, target_layer_idx=layer_idx
                )
                if hidden_states is None:
                    continue
                
                hidden_states = hidden_states.to(self.device)
                
                if hasattr(self.model, 'module'):
                    mlp = self.model.module.model.layers[layer_idx].mlp
                else:
                    mlp = self.model.model.layers[layer_idx].mlp
                
                with torch.no_grad():
                    current_output = mlp(hidden_states)
                    if isinstance(current_output, tuple):
                        current_output = current_output[0]
                
                baseline_output = self._expansion_baseline[layer_idx]
                if baseline_output.device != self.device:
                    baseline_output = baseline_output.to(self.device)
                
                cka_score = rep_evaluator._compute_linear_cka(
                    current_output.view(-1, current_output.size(-1)),
                    baseline_output.view(-1, baseline_output.size(-1))
                )
                
                results['layer_cka'][f'layer_{layer_idx}'] = {
                    'cka': cka_score,
                    'interpretation': 'high=stable, low=changed'
                }
                
                print_rank_0(f"   L{layer_idx:2d}: Layer CKA={cka_score:.4f} (vs baseline)", 0)
                
                del hidden_states, current_output
                self._clear_cuda_cache()
                
        except Exception as e:
            print_rank_0(f"[Warning] Layer CKA evaluation failed: {e}", 0)
        
        return results
    
    def _evaluate_expert_cka(
        self, 
        i_task: int, 
        layer_indices: list, 
        cached_batches: list
    ) -> dict:
        """
        Evaluate per-expert CKA (representation similarity between old and new experts).
        
        Returns:
            Dictionary with expert CKA results
        """
        results = {'cka': {}}
        
        print_rank_0(f"\n📊 Per-Expert CKA Evaluation...", 0)
        
        # Check what snapshots we have
        has_valid_cka_snapshots = (
            i_task in self.old_expert_snapshots and 
            len(self.old_expert_snapshots[i_task]) > 0
        )
        
        if not has_valid_cka_snapshots:
            print_rank_0(f"   [Skip] No expert snapshots available for CKA comparison", 0)
            return results
        
        try:
            # Count what's available
            frozen_count = 0
            trained_count = 0
            
            # Check first layer as representative
            if hasattr(self.model, 'module'):
                mlp = self.model.module.model.layers[layer_indices[0]].mlp
            else:
                mlp = self.model.model.layers[layer_indices[0]].mlp
            
            expert_range = self._get_experts_to_evaluate(i_task, mlp)
            total_with_snapshot = 0
            
            for expert_idx in expert_range:
                key = f"L{layer_indices[0]}_E{expert_idx}"
                if key in self.old_expert_snapshots[i_task]:
                    total_with_snapshot += 1
                    old_expert = self.old_expert_snapshots[i_task][key].to(self.device)
                    new_expert = mlp.scientific_experts[expert_idx]
                    weight_diff = self._compute_weight_difference(old_expert, new_expert)
                    old_expert.cpu()
                    del old_expert
                    
                    if weight_diff < 1e-10:
                        frozen_count += 1
                    else:
                        trained_count += 1
            
            # Print summary
            no_snapshot = len(list(expert_range)) - total_with_snapshot
            print_rank_0(f"   Summary for {len(list(expert_range))} experts:", 0)
            print_rank_0(f"   - With snapshots: {total_with_snapshot} experts", 0)
            print_rank_0(f"   - Frozen (weight diff < 1e-10): {frozen_count} → CKA=1.0", 0)
            print_rank_0(f"   - Trained (weight diff > 1e-10): {trained_count} → Computing CKA", 0)
            if no_snapshot > 0:
                print_rank_0(f"   - No snapshot: {no_snapshot} experts", 0)
            
            if trained_count == 0:
                print_rank_0(f"   [Info] All experts with snapshots are frozen (no training occurred)", 0)
                results['cka']['summary'] = {
                    'total_with_snapshot': total_with_snapshot,
                    'frozen_count': frozen_count,
                    'trained_count': trained_count,
                    'note': 'All experts frozen - CKA=1.0 for all'
                }
            else:
                # Compute CKA for experts that have changed
                print_rank_0(f"\n   Computing CKA for {trained_count} trained experts...", 0)
                
                for layer_idx in layer_indices:
                    if hasattr(self.model, 'module'):
                        mlp = self.model.module.model.layers[layer_idx].mlp
                    else:
                        mlp = self.model.model.layers[layer_idx].mlp
                    
                    hidden_states = self._prepare_inputs_for_metrics_from_cache(
                        cached_batches, max_samples=256, target_layer_idx=layer_idx
                    )
                    if hidden_states is None:
                        continue
                    
                    extractor = self._get_expert_extractor(self.cka_activation_source)
                    expert_range = self._get_experts_to_evaluate(i_task, mlp)
                    
                    for expert_idx in expert_range:
                        key = f"L{layer_idx}_E{expert_idx}"
                        
                        if key not in self.old_expert_snapshots[i_task]:
                            continue  # Skip new experts
                        
                        try:
                            old_expert = self.old_expert_snapshots[i_task][key].to(self.device)
                            new_expert = mlp.scientific_experts[expert_idx]
                            
                            weight_diff = self._compute_weight_difference(old_expert, new_expert)
                            if weight_diff < 1e-10:
                                results['cka'][key] = {
                                    'layer': layer_idx, 'expert': expert_idx,
                                    'score': 1.0, 'weight_diff': weight_diff,
                                    'note': 'frozen'
                                }
                                old_expert.cpu()
                                del old_expert
                                continue
                            
                            with MetricTimer(f"CKA {key}") as timer:
                                cka_computer = LinearCKA(
                                    mode=self.cka_mode,
                                    early_batches=self.cka_early_batches,
                                    device=self.device
                                )
                                
                                old_activations = extractor(old_expert, hidden_states)
                                new_activations = extractor(new_expert, hidden_states)
                                
                                cka_score = cka_computer.compute_cka_direct(
                                    old_activations, new_activations, verbose=False
                                )
                                
                                if cka_score < 0:
                                    cka_score = None
                                
                                del old_activations, new_activations
                            
                            results['cka'][key] = {
                                'layer': layer_idx, 'expert': expert_idx,
                                'score': cka_score if cka_score is not None else "failed",
                                'weight_diff': weight_diff,
                                'activation_source': self.cka_activation_source,
                                'time': timer.elapsed
                            }
                            
                            if cka_score is not None:
                                print_rank_0(f"   {key}: CKA={cka_score:.6f} (Δw={weight_diff:.2e})", 0)
                            
                        except Exception as e:
                            results['cka'][key] = {'score': None, 'error': str(e)}
                        finally:
                            if 'old_expert' in locals() and old_expert is not None:
                                old_expert.cpu()
                                del old_expert
                            self._clear_cuda_cache()
                    
                    del hidden_states
                    self._clear_cuda_cache()
                    
        except Exception as e:
            print_rank_0(f"[Warning] Expert CKA evaluation failed: {e}", 0)
            import traceback
            traceback.print_exc()
        
        return results
    
    def _evaluate_expert_flatness(
        self, 
        i_task: int, 
        task: str, 
        layer_indices: list, 
        cached_batches: list
    ) -> dict:
        """
        Evaluate expert flatness metrics (three-scenario and multi-scenario).
        
        Returns:
            Dictionary with flatness results
        """
        results = {}
        
        routing_mode = self.metric_routing_mode
        flatness_method = self.flatness_method
        
        # Three-scenario flatness evaluation for old/new experts
        print_rank_0(f"\n📈 Three-Scenario Flatness Evaluation (Old/New Experts)", 0)
        print_rank_0(f"   Scenario 1: Current task experts on current task data (baseline)", 0)
        print_rank_0(f"   Scenario 2: Current task experts on next task data (expansion indicator)", 0)
        print_rank_0(f"   Scenario 3: Next task's new experts on next task data (evaluation)", 0)
        
        if self.enable_multi_scenario_flatness:
            print_rank_0(f"\n📈 Multi-Scenario Flatness Evaluation (method={flatness_method}, routing={routing_mode})...", 0)
            print_rank_0(f"   Scenario 1: Current task experts on current task data (training sufficiency)", 0)
            print_rank_0(f"   Scenario 2: Current task experts on next task data (reusability w/o expansion)", 0)
            print_rank_0(f"   Scenario 3: Expanded experts on next task data (post-expansion readiness)", 0)
        else:
            print_rank_0(f"\n📈 Flatness Evaluation (method={flatness_method}, routing={routing_mode})...", 0)
        
        try:
            for layer_idx in layer_indices:
                print_rank_0(f"\n   Layer {layer_idx}:", 0)
                
                if hasattr(self.model, 'module'):
                    mlp = self.model.module.model.layers[layer_idx].mlp
                else:
                    mlp = self.model.model.layers[layer_idx].mlp
                
                # Prepare hidden states for flatness using cached batches
                all_hidden_states = self._prepare_inputs_for_metrics_from_cache(
                    cached_batches, 
                    max_samples=self.flatness_max_samples,
                    target_layer_idx=layer_idx
                )
                
                if all_hidden_states is None:
                    print_rank_0(f"   [Warning] Could not prepare hidden states for layer {layer_idx}", 0)
                    continue
                
                # Determine which experts to evaluate
                expert_range = self._get_experts_to_evaluate(i_task, mlp)
                
                # Identify old and new experts
                current_task_expert_range = self.task2expert_range.get(i_task, range(0))
                old_expert_indices = []
                new_expert_indices = []
                
                # Old experts: experts from previous tasks
                for prev_task_id in range(i_task):
                    if prev_task_id in self.task2expert_range:
                        prev_range = self.task2expert_range[prev_task_id]
                        old_expert_indices.extend(list(prev_range))
                
                # New experts: experts from current task
                new_expert_indices = list(current_task_expert_range)
                
                # Trainable experts for Scenario 1 & 2: CURRENT task's experts
                trainable_expert_indices = list(current_task_expert_range)
                
                print_rank_0(f"   Task {i_task}: Current task experts: {list(current_task_expert_range)}", 0)
                print_rank_0(f"   Old experts: {old_expert_indices} (total: {len(old_expert_indices)})", 0)
                print_rank_0(f"   New experts: {new_expert_indices} (total: {len(new_expert_indices)})", 0)
                print_rank_0(f"   Trainable experts for flatness: {trainable_expert_indices} (total: {len(trainable_expert_indices)})", 0)
                
                # Get routed tokens for all experts if using routing-aware mode
                print_rank_0(f"   Computing routing for {len(all_hidden_states)} tokens...", 0)
                flatness_routed_tokens_dict = self._get_routed_tokens_if_needed(
                    all_hidden_states, mlp, expert_range, routing_mode
                )
                
                # Three-Scenario Flatness Evaluation
                if len(trainable_expert_indices) > 0:
                    # Scenario 1: Current task experts on current task data
                    expert_range_str = f"E{trainable_expert_indices[0]}-{trainable_expert_indices[-1]}" if len(trainable_expert_indices) > 1 else f"E{trainable_expert_indices[0]}"
                    print_rank_0(f"\n   [Three-Scenario] Scenario 1: {expert_range_str} on task {i_task} ({task})", 0)
                    
                    scenario1_cached_batches = cached_batches
                    scenario1_hidden_states = all_hidden_states
                    
                    if scenario1_hidden_states is not None:
                        scenario1_routed_tokens_dict = self._get_routed_tokens_if_needed(
                            scenario1_hidden_states, mlp, trainable_expert_indices, routing_mode
                        )
                        
                        # Evaluate trainable experts on their task data
                        for expert_idx in trainable_expert_indices[:10]:  # Limit to first 10
                            if expert_idx >= len(mlp.scientific_experts):
                                continue
                            
                            expert = mlp.scientific_experts[expert_idx]
                            key = f"L{layer_idx}_E{expert_idx}"
                            
                            # Get routed tokens
                            if routing_mode == 'routed' and scenario1_routed_tokens_dict is not None:
                                hidden_states_scenario1, routing_stats = scenario1_routed_tokens_dict.get(expert_idx, (scenario1_hidden_states, None))
                            else:
                                hidden_states_scenario1 = scenario1_hidden_states
                                routing_stats = None
                            
                            # Compute flatness
                            if i_task == 0:
                                scenario1_result = self._evaluate_expert_flatness_scenario(
                                    expert, expert_idx, layer_idx,
                                    hidden_states_scenario1, scenario1_cached_batches,
                                    'scenario1_current_expert_current_task',
                                    f'Current task expert E{expert_idx} on current task T{i_task}'
                                )
                            else:
                                scenario1_result = self._evaluate_expert_flatness_scenario_trainable_only(
                                    expert, expert_idx, layer_idx,
                                    hidden_states_scenario1, scenario1_cached_batches,
                                    'scenario1_old_expert_old_task',
                                    f'Old expert E{expert_idx} on old task T{i_task}'
                                )
                            scenario1_result['routing_stats'] = routing_stats
                            scenario1_result['task_id'] = i_task
                            scenario1_result['task_name'] = task
                            
                            scenario1_key = f"E{trainable_expert_indices[0]}-{trainable_expert_indices[-1]}_on_task_{i_task}_flatness" if len(trainable_expert_indices) > 1 else f"E{trainable_expert_indices[0]}_on_task_{i_task}_flatness"
                            if scenario1_key not in results:
                                results[scenario1_key] = {}
                            results[scenario1_key][key] = scenario1_result
                            
                            if 'old_expert_old_task_flatness' not in results:
                                results['old_expert_old_task_flatness'] = {}
                            results['old_expert_old_task_flatness'][key] = scenario1_result
                    
                    # Scenario 2: Trainable experts on next task data
                    next_task_idx = i_task + 1
                    next_task_exists = next_task_idx < len(self.train_task_list)
                    
                    if next_task_exists:
                        next_task_name = self.train_task_list[next_task_idx]
                        expert_range_str = f"E{trainable_expert_indices[0]}-{trainable_expert_indices[-1]}" if len(trainable_expert_indices) > 1 else f"E{trainable_expert_indices[0]}"
                        print_rank_0(f"\n   [Three-Scenario] Scenario 2: {expert_range_str} on task {next_task_idx} ({next_task_name})", 0)
                        
                        next_task_dataloader = self.eval_task_list.get(next_task_name) if hasattr(self, 'eval_task_list') else None
                        if next_task_dataloader is None:
                            next_task_dataloader = self._create_task_dataloader(next_task_name)
                            if next_task_dataloader is not None:
                                if not hasattr(self, 'eval_task_list'):
                                    self.eval_task_list = {}
                                self.eval_task_list[next_task_name] = next_task_dataloader
                        
                        if next_task_dataloader is not None:
                            next_task_cached_batches = self._cache_dataloader_batches(
                                next_task_dataloader,
                                num_batches=min(10, self.metric_num_batches)
                            )
                            
                            next_task_hidden_states = self._prepare_inputs_for_metrics_from_cache(
                                next_task_cached_batches,
                                max_samples=self.flatness_max_samples,
                                target_layer_idx=layer_idx
                            )
                            
                            if next_task_hidden_states is not None:
                                scenario2_routed_tokens_dict = self._get_routed_tokens_if_needed(
                                    next_task_hidden_states, mlp, trainable_expert_indices, routing_mode
                                )
                                
                                for expert_idx in trainable_expert_indices[:10]:
                                    if expert_idx >= len(mlp.scientific_experts):
                                        continue
                                    
                                    expert = mlp.scientific_experts[expert_idx]
                                    key = f"L{layer_idx}_E{expert_idx}"
                                    
                                    if routing_mode == 'routed' and scenario2_routed_tokens_dict is not None:
                                        hidden_states_scenario2, routing_stats = scenario2_routed_tokens_dict.get(expert_idx, (next_task_hidden_states, None))
                                    else:
                                        hidden_states_scenario2 = next_task_hidden_states
                                        routing_stats = None
                                    
                                    if i_task == 0:
                                        scenario2_result = self._evaluate_expert_flatness_scenario(
                                            expert, expert_idx, layer_idx,
                                            hidden_states_scenario2, next_task_cached_batches,
                                            'scenario2_current_expert_next_task',
                                            f'Current task expert E{expert_idx} on next task T{next_task_idx}'
                                        )
                                    else:
                                        scenario2_result = self._evaluate_expert_flatness_scenario_trainable_only(
                                            expert, expert_idx, layer_idx,
                                            hidden_states_scenario2, next_task_cached_batches,
                                            'scenario2_old_expert_new_task',
                                            f'Old expert E{expert_idx} on new task T{i_task}'
                                        )
                                    scenario2_result['routing_stats'] = routing_stats
                                    scenario2_result['current_task_id'] = i_task
                                    scenario2_result['next_task_id'] = next_task_idx
                                    scenario2_result['next_task_name'] = next_task_name
                                    
                                    scenario2_key = f"E{trainable_expert_indices[0]}-{trainable_expert_indices[-1]}_on_task_{next_task_idx}_flatness" if len(trainable_expert_indices) > 1 else f"E{trainable_expert_indices[0]}_on_task_{next_task_idx}_flatness"
                                    if scenario2_key not in results:
                                        results[scenario2_key] = {}
                                    results[scenario2_key][key] = scenario2_result
                                    
                                    if 'trainable_expert_next_task_flatness' not in results:
                                        results['trainable_expert_next_task_flatness'] = {}
                                    results['trainable_expert_next_task_flatness'][key] = scenario2_result
                    
                    # Scenario 3: Next task's new experts on next task data
                    if next_task_exists:
                        next_task_expert_range = self.task2expert_range.get(next_task_idx, range(0))
                        next_task_new_expert_indices = list(next_task_expert_range)
                        
                        if len(next_task_new_expert_indices) > 0:
                            next_expert_range_str = f"E{next_task_new_expert_indices[0]}-{next_task_new_expert_indices[-1]}" if len(next_task_new_expert_indices) > 1 else f"E{next_task_new_expert_indices[0]}"
                            print_rank_0(f"\n   [Three-Scenario] Scenario 3: {next_expert_range_str} on task {next_task_idx} ({next_task_name})", 0)
                            
                            if next_task_dataloader is not None:
                                if next_task_hidden_states is not None:
                                    existing_next_task_expert_indices = [idx for idx in next_task_new_expert_indices if idx < len(mlp.scientific_experts)]
                                    if len(existing_next_task_expert_indices) > 0:
                                        next_task_new_expert_routed_tokens_dict = self._get_routed_tokens_if_needed(
                                            next_task_hidden_states, mlp, existing_next_task_expert_indices, routing_mode
                                        )
                                    else:
                                        next_task_new_expert_routed_tokens_dict = None
                                    
                                    for expert_idx in next_task_new_expert_indices:
                                        if expert_idx >= len(mlp.scientific_experts):
                                            continue
                                        
                                        expert = mlp.scientific_experts[expert_idx]
                                        key = f"L{layer_idx}_E{expert_idx}"
                                        
                                        if routing_mode == 'routed' and next_task_new_expert_routed_tokens_dict is not None:
                                            hidden_states_scenario3, routing_stats = next_task_new_expert_routed_tokens_dict.get(expert_idx, (next_task_hidden_states, None))
                                        else:
                                            hidden_states_scenario3 = next_task_hidden_states
                                            routing_stats = None
                                        
                                        scenario3_result = self._evaluate_expert_flatness_scenario(
                                            expert, expert_idx, layer_idx,
                                            hidden_states_scenario3, next_task_cached_batches,
                                            'scenario3_next_task_new_expert_next_task',
                                            f'Next task T{next_task_idx} new expert E{expert_idx} on next task data'
                                        )
                                        scenario3_result['routing_stats'] = routing_stats
                                        scenario3_result['next_task_id'] = next_task_idx
                                        scenario3_result['next_task_name'] = next_task_name
                                        
                                        scenario3_key = f"E{next_task_new_expert_indices[0]}-{next_task_new_expert_indices[-1]}_on_task_{next_task_idx}_flatness" if len(next_task_new_expert_indices) > 1 else f"E{next_task_new_expert_indices[0]}_on_task_{next_task_idx}_flatness"
                                        if scenario3_key not in results:
                                            results[scenario3_key] = {}
                                        results[scenario3_key][key] = scenario3_result
                                        
                                        if 'new_expert_new_task_flatness' not in results:
                                            results['new_expert_new_task_flatness'] = {}
                                        results['new_expert_new_task_flatness'][key] = scenario3_result
                
                # Standard flatness evaluation (if multi-scenario is disabled)
                if not self.enable_multi_scenario_flatness:
                    for expert_idx in expert_range:
                        if expert_idx >= len(mlp.scientific_experts):
                            continue
                        
                        key = f"L{layer_idx}_E{expert_idx}"
                        
                        try:
                            expert = mlp.scientific_experts[expert_idx]
                            
                            # Select tokens based on routing mode
                            routing_stats = None
                            if routing_mode == 'routed' and flatness_routed_tokens_dict is not None:
                                routed_data = flatness_routed_tokens_dict.get(expert_idx)
                                if routed_data is not None:
                                    hidden_states, routing_stats = routed_data
                                else:
                                    hidden_states = all_hidden_states
                            else:
                                hidden_states = all_hidden_states
                            
                            if hidden_states is None or len(hidden_states) == 0:
                                print_rank_0(f"   [Error] No tokens available for expert {expert_idx}, skipping", 0)
                                results['flatness'][key] = {
                                    'error': 'No tokens available',
                                    'layer': layer_idx,
                                    'expert': expert_idx
                                }
                                continue
                            
                            # Limit samples
                            if len(hidden_states) > self.flatness_max_samples:
                                hidden_states = hidden_states[:self.flatness_max_samples]
                            
                            expert_results = self._evaluate_expert_flatness_scenario(
                                expert, expert_idx, layer_idx,
                                hidden_states, cached_batches,
                                'standard_flatness',
                                f'Expert E{expert_idx} flatness'
                            )
                            expert_results['routing_stats'] = routing_stats
                            
                            if 'flatness' not in results:
                                results['flatness'] = {}
                            results['flatness'][key] = expert_results
                            
                        except Exception as e:
                            print_rank_0(f"   [Warning] Flatness evaluation failed for {key}: {e}", 0)
                            if 'flatness' not in results:
                                results['flatness'] = {}
                            results['flatness'][key] = {'error': str(e)}
                
                del all_hidden_states
                self._clear_cuda_cache()
                
        except Exception as e:
            print_rank_0(f"[Warning] Expert flatness evaluation failed: {e}", 0)
            import traceback
            traceback.print_exc()
        
        return results
    
    def _print_metrics_summary(self, results: dict):
        """Print a summary of the metric results with ASCII visualization."""
        print_rank_0(f"\n📋 Metrics Summary:", 0)
        
        # CKA summary
        if results['cka']:
            cka_scores = [v['score'] for v in results['cka'].values() 
                         if isinstance(v.get('score'), (int, float))]
            if cka_scores:
                print_rank_0(f"   CKA (activation={self.cka_activation_source}):", 0)
                print_rank_0(f"      Mean: {sum(cka_scores)/len(cka_scores):.4f}", 0)
                print_rank_0(f"      Min: {min(cka_scores):.4f}, Max: {max(cka_scores):.4f}", 0)
        
        # Flatness summary by layer with ASCII visualization
        # Print both initial and trained flatness if available
        for flatness_key, flatness_label in [('flatness_initial', 'Initial (after expansion)'), 
                                             ('flatness_trained', 'Trained (after training)')]:
            if results.get(flatness_key):
                flatness_by_layer = {}
                for key, v in results[flatness_key].items():
                    if 'error' in v:
                        continue
                    layer = v.get('layer', 0)
                    if layer not in flatness_by_layer:
                        flatness_by_layer[layer] = []
                    flatness_by_layer[layer].append(v)
                
                if flatness_by_layer:
                    print_rank_0(f"   Flatness {flatness_label} (method={self.flatness_method}, "
                               f"loss={self.flatness_loss_type}, routing={self.metric_routing_mode}):", 0)
                    for layer, experts in sorted(flatness_by_layer.items()):
                        if self.flatness_method == 'hessian':
                            eigenvalues = [e.get('hessian_top_eigenvalue', 0.0) for e in experts]
                            traces = [e.get('hessian_trace', 0.0) for e in experts]
                            print_rank_0(f"      Layer {layer}: λ_max avg={sum(eigenvalues)/len(eigenvalues):.4e}, "
                                       f"trace avg={sum(traces)/len(traces):.4e}", 0)
                        elif self.flatness_method == 'landscape':
                            max_changes = [e.get('landscape_max_loss_change', 0.0) for e in experts]
                            print_rank_0(f"      Layer {layer}: max_loss_change avg={sum(max_changes)/len(max_changes):.4e}", 0)
                    
                    # ASCII visualization of flatness across layers
                    if flatness_by_layer and self.flatness_method == 'hessian':
                        self._print_flatness_visualization(flatness_by_layer, label=flatness_label)
        
        # Also print main flatness dict for backward compatibility
        if results['flatness']:
            flatness_by_layer = {}
            for key, v in results['flatness'].items():
                if 'error' in v:
                    continue
                layer = v.get('layer', 0)
                if layer not in flatness_by_layer:
                    flatness_by_layer[layer] = []
                flatness_by_layer[layer].append(v)
            
            if flatness_by_layer:
                print_rank_0(f"   Flatness Summary (method={self.flatness_method}, routing={self.metric_routing_mode}):", 0)
                for layer, experts in sorted(flatness_by_layer.items()):
                    if self.flatness_method == 'hessian':
                        eigenvalues = [e.get('hessian_top_eigenvalue', 0.0) for e in experts]
                        traces = [e.get('hessian_trace', 0.0) for e in experts]
                        print_rank_0(f"      Layer {layer}: λ_max avg={sum(eigenvalues)/len(eigenvalues):.4e}, "
                                   f"trace avg={sum(traces)/len(traces):.4e}", 0)
    
    def _print_flatness_visualization(self, flatness_by_layer: dict, label: str = ""):
        """
        Print ASCII bar chart and generate matplotlib plot of flatness (λ_max) across layers.
        """
        label_str = f" ({label})" if label else ""
        print_rank_0(f"\n   📊 Flatness Visualization{label_str} (log scale λ_max):", 0)
        print_rank_0(f"      ─────────────────────────────────────────────────────", 0)
        
        # Collect λ_max averages per layer
        layer_eigenvalues = {}
        for layer, experts in sorted(flatness_by_layer.items()):
            eigenvalues = [e.get('hessian_top_eigenvalue', 0.0) for e in experts if e.get('hessian_top_eigenvalue', 0.0) > 0]
            if eigenvalues:
                layer_eigenvalues[layer] = sum(eigenvalues) / len(eigenvalues)
        
        if not layer_eigenvalues:
            return
        
        # Use log scale for visualization
        import math
        log_values = {k: math.log10(max(v, 1e-15)) for k, v in layer_eigenvalues.items()}
        min_log = min(log_values.values())
        max_log = max(log_values.values())
        
        # Normalize to bar width (max 40 chars)
        bar_width = 40
        
        for layer in sorted(layer_eigenvalues.keys()):
            val = layer_eigenvalues[layer]
            log_val = log_values[layer]
            
            # Normalize bar length
            if max_log > min_log:
                bar_len = int((log_val - min_log) / (max_log - min_log) * bar_width)
            else:
                bar_len = bar_width // 2
            bar_len = max(1, bar_len)  # At least 1 char
            
            # Create bar with gradient
            bar = "█" * min(bar_len, bar_width)
            
            # Color indicator based on magnitude
            if val < 1e-9:
                indicator = "🟢"  # Very flat (good)
            elif val < 1e-6:
                indicator = "🟡"  # Moderately flat
            elif val < 1e-3:
                indicator = "🟠"  # Some sharpness
            else:
                indicator = "🔴"  # Sharp (potentially problematic)
            
            print_rank_0(f"      L{layer:2d} {indicator} |{bar:<{bar_width}}| {val:.2e}", 0)
        
        print_rank_0(f"      ─────────────────────────────────────────────────────", 0)
        print_rank_0(f"      Legend: 🟢 flat (<1e-9) 🟡 moderate 🟠 sharp 🔴 very sharp", 0)
        
        # Generate matplotlib plot (not just text)
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            layers = sorted(layer_eigenvalues.keys())
            values = [layer_eigenvalues[l] for l in layers]
            
            # Create bar chart with log scale
            colors = []
            for val in values:
                if val < 1e-9:
                    colors.append('green')
                elif val < 1e-6:
                    colors.append('yellow')
                elif val < 1e-3:
                    colors.append('orange')
                else:
                    colors.append('red')
            
            bars = ax.bar(layers, values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_yscale('log')
            ax.set_xlabel('Layer Index', fontsize=12)
            ax.set_ylabel('λ_max (Hessian Top Eigenvalue)', fontsize=12)
            ax.set_title(f'Flatness Visualization{label_str} (log scale)', fontsize=14)
            ax.grid(True, alpha=0.3, which='both')
            
            # Add value labels on bars
            for i, (layer, val) in enumerate(zip(layers, values)):
                ax.text(layer, val, f'{val:.2e}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            plt.tight_layout()
            
            # Save plot
            if hasattr(self, 'args') and hasattr(self.args, 'output_dir'):
                from pathlib import Path
                output_dir = Path(self.args.output_dir) / 'expert_metrics'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Create safe filename from label
                safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').lower() if label else 'flatness'
                filename = f'flatness_visualization_{safe_label}.png'
                filepath = output_dir / filename
                
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                print_rank_0(f"      📈 Plot saved to: {filepath}", 0)
            
            plt.close()
            
        except ImportError:
            print_rank_0(f"      [Info] Matplotlib not available, skipping plot generation", 0)
        except Exception as e:
            print_rank_0(f"      [Warning] Failed to generate plot: {e}", 0)
    
    def _create_task_dataloader(self, task_name: str):
        """
        Dynamically create a dataloader for a given task name.
        
        Args:
            task_name: Name of the task/dataset
            
        Returns:
            DataLoader for the task, or None if creation fails
        """
        try:
            # Check if we have the necessary attributes
            if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                print_rank_0(f"   [Warning] Tokenizer not available, cannot create dataloader for {task_name}", 0)
                return None
            
            if not hasattr(self, 'args') or not hasattr(self.args, 'data_path'):
                print_rank_0(f"   [Warning] Data path not available, cannot create dataloader for {task_name}", 0)
                return None
            
            # Import required modules
            from utils.data.data_utils import create_prompt_dataset
            from utils.data.data_collator import DataCollator
            from torch.utils.data import DataLoader
            import os
            
            # Create dataset
            dataset_path = os.path.join(self.args.data_path, task_name)
            if not os.path.exists(dataset_path):
                print_rank_0(f"   [Warning] Dataset path not found: {dataset_path}", 0)
                return None
            
            _, _, eval_dataset = create_prompt_dataset(
                local_rank=getattr(self.args, 'local_rank', -1),
                data_path=dataset_path,
                output_path="/tmp/eval_data",
                seed=getattr(self.args, 'seed', 1234),
                distributed=False
            )
            
            # Create collator
            collator = DataCollator(
                self.tokenizer,
                model=self.model if hasattr(self, 'model') else None,
                max_prompt_len=getattr(self.args, 'max_prompt_len', 2048),
                max_ans_len=getattr(self.args, 'max_ans_len', 512),
                inference=False
            )
            
            # Create dataloader
            batch_size = getattr(self.args, 'batch_size', 16)
            dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                collate_fn=collator,
                shuffle=False
            )
            
            print_rank_0(f"   [Info] Created dataloader for task {task_name}", 0)
            return dataloader
            
        except Exception as e:
            print_rank_0(f"   [Warning] Failed to create dataloader for {task_name}: {e}", 0)
            import traceback
            traceback.print_exc()
            return None
    
    def _create_expert_dataloader(self, hidden_states: torch.Tensor, batch_size: int = 32):
        """
        Create a simple dataloader from hidden states for expert evaluation.
        
        Args:
            hidden_states: Tensor of shape (num_samples, hidden_dim)
            batch_size: Batch size
            
        Returns:
            Generator yielding batches
        """
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataset = SimpleDataset(hidden_states)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    def _compute_flatness_metrics(
        self,
        expert: nn.Module,
        loss_fn: Callable,
        expert_dataloader: torch.utils.data.DataLoader,
        flatness_method: str,
        scenario_name: str,
        trainable_only: bool = False,
        trainable_param_names: Optional[List[str]] = None,
        use_landscape_fallback: bool = False
    ) -> dict:
        """
        Unified method to compute flatness metrics (Hessian or Landscape).
        
        IMPORTANT: This method preserves the ability to compute Hessian on real task loss.
        When loss_type='final', the loss_fn uses full_model forward pass to compute actual
        task loss, and the Hessian is computed with respect to expert parameters.
        The expert_dataloader may contain hidden_states, but final_loss_fn will use
        model_input_batches (passed during loss_fn creation) to run full_model forward pass.
        
        Args:
            expert: Expert module to evaluate
            loss_fn: Loss function (can be final_loss_fn for real task loss computation)
            expert_dataloader: DataLoader for evaluation (may contain hidden_states for non-final loss,
                             but final_loss_fn uses model_input_batches internally)
            flatness_method: 'hessian' or 'landscape'
            scenario_name: Scenario name for logging
            trainable_only: If True, only evaluate trainable parameters
            trainable_param_names: List of trainable parameter names (if trainable_only=True)
            use_landscape_fallback: Whether this is a fallback evaluation
            
        Returns:
            Dictionary with flatness metrics
        """
        results = {}
        
        try:
            if flatness_method == 'hessian' and not use_landscape_fallback:
                # Temporarily enable requires_grad
                original_requires_grad = {}
                for name, param in expert.named_parameters():
                    original_requires_grad[name] = param.requires_grad
                    if trainable_only:
                        # Only enable grad for trainable params
                        param.requires_grad = name in (trainable_param_names or [])
                    else:
                        param.requires_grad = True
                
                # Hessian Top Eigenvalue
                with MetricTimer("Hessian Eigenvalue" + (" (Trainable Only)" if trainable_only else "")) as timer:
                    try:
                        if trainable_only:
                            # Create custom Hessian analyzer for trainable-only
                            class TrainableOnlyHessianSpectrum(HessianSpectrum):
                                def __init__(self, model, loss_fn, device, power_iterations, trainable_param_names):
                                    super().__init__(model, loss_fn, device, power_iterations)
                                    self.trainable_param_names = set(trainable_param_names)
                                
                                def _get_params(self):
                                    params = []
                                    for name, param in self.model.named_parameters():
                                        if name in self.trainable_param_names and param.requires_grad:
                                            params.append(param)
                                    return params
                            
                            hessian_analyzer = TrainableOnlyHessianSpectrum(
                                expert, loss_fn, self.device, self.power_iterations,
                                trainable_param_names or []
                            )
                        else:
                            # NOTE: loss_fn can be final_loss_fn which computes real task loss
                            # by running full_model forward pass using model_input_batches.
                            # The expert_dataloader may contain hidden_states, but final_loss_fn
                            # will use model_input_batches internally to compute actual task loss.
                            # This ensures Hessian is computed w.r.t. real task loss.
                            hessian_analyzer = HessianSpectrum(
                                expert, loss_fn, self.device, self.power_iterations
                            )
                        
                        num_batches_for_hessian = min(self.metric_num_batches, 10)
                        # Compute Hessian top eigenvalue
                        # For final_loss_fn: computes Hessian of expert params w.r.t. real task loss
                        # For other loss types: computes Hessian w.r.t. reconstruction/expert loss
                        lambda_max, _ = hessian_analyzer.compute_top_eigenvalue(
                            dataloader=expert_dataloader,
                            num_batches=num_batches_for_hessian
                        )
                    except Exception as e:
                        print_rank_0(f"   [{scenario_name}] [Warning] Hessian failed: {e}", 0)
                        lambda_max = 0.0
                
                results['hessian_top_eigenvalue'] = float(lambda_max)
                results['hessian_eigenvalue_time'] = timer.elapsed
                
                # Epsilon-Sharpness
                eps_sharpness = EpsilonSharpness(epsilon=self.sharpness_epsilon)
                sharpness_score = eps_sharpness.compute(lambda_max)
                results['epsilon_sharpness'] = float(sharpness_score)
                results['epsilon'] = self.sharpness_epsilon
                
                # Hessian Diagonal
                with MetricTimer("Hessian Diagonal" + (" (Trainable Only)" if trainable_only else "")) as timer:
                    try:
                        if trainable_only:
                            class TrainableOnlyHessianDiagonal(HessianDiagonal):
                                def __init__(self, model, loss_fn, device, num_samples, trainable_param_names):
                                    super().__init__(model, loss_fn, device, num_samples)
                                    self.trainable_param_names = set(trainable_param_names)
                                
                                def _get_params(self):
                                    params = []
                                    for name, param in self.model.named_parameters():
                                        if name in self.trainable_param_names and param.requires_grad:
                                            params.append(param)
                                    return params
                            
                            hessian_diag = TrainableOnlyHessianDiagonal(
                                expert, loss_fn, self.device, self.hutchinson_samples,
                                trainable_param_names or []
                            )
                        else:
                            # NOTE: loss_fn can be final_loss_fn which computes real task loss
                            # by running full_model forward pass using model_input_batches.
                            # This ensures Hessian trace is computed w.r.t. real task loss.
                            hessian_diag = HessianDiagonal(
                                expert, loss_fn, self.device, self.hutchinson_samples
                            )
                        
                        num_batches_for_trace = min(self.metric_num_batches, 10)
                        # Compute Hessian diagonal trace
                        # For final_loss_fn: computes trace of Hessian w.r.t. real task loss
                        # For other loss types: computes trace w.r.t. reconstruction/expert loss
                        diag_result, _ = hessian_diag.compute_trace(
                            dataloader=expert_dataloader,
                            num_batches=num_batches_for_trace
                        )
                    except Exception as e:
                        print_rank_0(f"   [{scenario_name}] [Warning] Hessian diagonal failed: {e}", 0)
                        diag_result = {'trace': 0.0, 'mean_diagonal': 0.0}
                
                results['hessian_trace'] = float(diag_result['trace'])
                results['hessian_mean_diagonal'] = float(diag_result['mean_diagonal'])
                results['hessian_diagonal_time'] = timer.elapsed
                
                trainable_note = " (trainable only)" if trainable_only else ""
                print_rank_0(f"   [{scenario_name}] λ_max={lambda_max:.6e}, trace={diag_result['trace']:.6e}{trainable_note}", 0)
                
                # Restore requires_grad
                for name, param in expert.named_parameters():
                    if name in original_requires_grad:
                        param.requires_grad = original_requires_grad[name]
            
            elif flatness_method == 'landscape' or use_landscape_fallback:
                fallback_note = " (fallback: no routed tokens)" if use_landscape_fallback else ""
                trainable_note = " (Trainable Only)" if trainable_only else ""
                with MetricTimer(f"Landscape Flatness{fallback_note}{trainable_note}") as timer:
                    try:
                        if trainable_only:
                            # Create wrapper for trainable-only landscape
                            class TrainableOnlyLandscapeFlatness(LandscapeFlatness):
                                def __init__(self, model, loss_fn, device, steps, multiplier, num_directions, norm_type, trainable_param_names):
                                    super().__init__(model, loss_fn, device, steps, multiplier, num_directions, norm_type)
                                    self.trainable_param_names = set(trainable_param_names)
                                
                                def _generate_random_direction(self, original_weights):
                                    direction = {}
                                    for name, weight in original_weights.items():
                                        if name in self.trainable_param_names:
                                            direction[name] = torch.randn_like(weight)
                                        else:
                                            direction[name] = torch.zeros_like(weight)
                                    return direction
                            
                            landscape_analyzer = TrainableOnlyLandscapeFlatness(
                                expert, loss_fn, self.device, self.landscape_steps,
                                self.landscape_multiplier, self.landscape_num_directions,
                                "layer", trainable_param_names or []
                            )
                        else:
                            landscape_analyzer = LandscapeFlatness(
                                expert, loss_fn, self.device, self.landscape_steps,
                                self.landscape_multiplier, self.landscape_num_directions,
                                norm_type="layer"
                            )
                        
                        num_batches_for_landscape = min(self.metric_num_batches, 5)
                        landscape_result, _ = landscape_analyzer.compute(
                            dataloader=expert_dataloader,
                            num_batches=num_batches_for_landscape
                        )
                    except Exception as e:
                        print_rank_0(f"   [{scenario_name}] [Warning] Landscape failed: {e}", 0)
                        landscape_result = {
                            'max_loss_change': 0.0,
                            'avg_loss_change': 0.0,
                            'loss_curvature': 0.0,
                            'visage_score': 0.0
                        }
                
                results['landscape_max_loss_change'] = float(landscape_result.get('max_loss_change', 0.0))
                results['landscape_avg_loss_change'] = float(landscape_result.get('avg_loss_change', 0.0))
                results['landscape_curvature'] = float(landscape_result.get('loss_curvature', 0.0))
                results['landscape_visage_score'] = float(landscape_result.get('visage_score', 0.0))
                results['landscape_time'] = timer.elapsed
                
                if use_landscape_fallback:
                    results['fallback_method'] = 'landscape'
                    results['fallback_reason'] = 'no_routed_tokens'
                    trainable_note = " (trainable only)" if trainable_only else ""
                    print_rank_0(f"   [{scenario_name}] VISAGE={landscape_result.get('visage_score', 0.0):.6e} (fallback: no routed tokens){trainable_note}", 0)
                else:
                    trainable_note = " (trainable only)" if trainable_only else ""
                    print_rank_0(f"   [{scenario_name}] VISAGE={landscape_result.get('visage_score', 0.0):.6e}{trainable_note}", 0)
            
            else:
                print_rank_0(f"   [{scenario_name}] [Warning] Unknown flatness method: {flatness_method}", 0)
        
        except Exception as e:
            print_rank_0(f"   [{scenario_name}] [Error] Flatness computation failed: {e}", 0)
            results['error'] = str(e)
        
        return results
    
    def _evaluate_expert_flatness_scenario(
        self,
        expert: nn.Module,
        expert_idx: int,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cached_batches: list,
        scenario_name: str,
        scenario_description: str
    ) -> dict:
        """
        Evaluate flatness for a single expert in a specific scenario.
        
        Args:
            expert: The expert module to evaluate
            expert_idx: Expert index
            layer_idx: Layer index
            hidden_states: Hidden states tensor for this expert
            cached_batches: Cached input batches (for final loss)
            scenario_name: Name of the scenario (e.g., 'current_task_current_data')
            scenario_description: Description of the scenario
            
        Returns:
            Dictionary with flatness metrics
        """
        key = f"L{layer_idx}_E{expert_idx}"
        routing_mode = self.metric_routing_mode
        flatness_method = self.flatness_method
        
        print_rank_0(f"   [{scenario_name}] {key}: {scenario_description}", 0)
        
        # Check if we have routed tokens or need fallback
        use_landscape_fallback = False
        if hidden_states is None:
            # No routed tokens available
            if self.use_landscape_fallback:
                # Use landscape method as fallback (doesn't require specific tokens)
                print_rank_0(f"   [{scenario_name}] [Info] No routed tokens, using landscape method as fallback", 0)
                use_landscape_fallback = True
                flatness_method = 'landscape'  # Override method for fallback
                # Use all hidden states for landscape evaluation (if available from cached_batches)
                # We'll create a dataloader from cached_batches instead
                hidden_states = None  # Will be handled in landscape evaluation
            else:
                # Skip evaluation if fallback is disabled
                print_rank_0(f"   [{scenario_name}] [Skip] No routed tokens and landscape fallback disabled", 0)
                return {
                    'skipped': True,
                    'skip_reason': 'no_routed_tokens',
                    'scenario': scenario_name,
                    'expert': expert_idx,
                    'layer': layer_idx
                }
        
        # Validate hidden states (if not using fallback)
        if not use_landscape_fallback:
            actual_hidden_states_len = len(hidden_states) if hidden_states is not None else 0
            if actual_hidden_states_len == 0:
                print_rank_0(f"   [{scenario_name}] [Error] No tokens available, skipping", 0)
                return {'error': 'No tokens available', 'scenario': scenario_name}
            num_tokens = actual_hidden_states_len
        else:
            # For landscape fallback, we'll use all available tokens from cached_batches
            num_tokens = 0  # Will be determined during landscape evaluation
        
        # Limit samples to max_samples if needed
        if num_tokens > self.flatness_max_samples:
            hidden_states = hidden_states[:self.flatness_max_samples]
            num_tokens = self.flatness_max_samples
        
        expert_results = {
            'layer': layer_idx,
            'expert': expert_idx,
            'scenario': scenario_name,
            'scenario_description': scenario_description,
            'routing_mode': routing_mode,
            'flatness_method': flatness_method,
            'loss_type': self.flatness_loss_type,
            'num_samples': num_tokens
        }
        
        # Handle landscape fallback: prepare hidden states from cached_batches
        if use_landscape_fallback:
            if cached_batches and len(cached_batches) > 0:
                # Prepare hidden states from cached batches for landscape evaluation
                all_hidden_states_list = []
                for batch in cached_batches[:min(5, len(cached_batches))]:  # Use first 5 batches
                    batch_hidden_states = self._prepare_inputs_for_metrics_from_cache(
                        [batch], max_samples=128, target_layer_idx=layer_idx
                    )
                    if batch_hidden_states is not None:
                        all_hidden_states_list.append(batch_hidden_states)
                
                if all_hidden_states_list:
                    hidden_states = torch.cat(all_hidden_states_list, dim=0)
                    num_tokens = len(hidden_states)
                    expert_results['num_samples'] = num_tokens
                    expert_results['fallback_method'] = 'landscape'
                    expert_results['fallback_reason'] = 'no_routed_tokens'
                else:
                    print_rank_0(f"   [{scenario_name}] [Error] Could not prepare hidden states for landscape fallback", 0)
                    return {'error': 'Could not prepare hidden states for fallback', 'scenario': scenario_name}
            else:
                print_rank_0(f"   [{scenario_name}] [Error] No cached batches available for landscape fallback", 0)
                return {'error': 'No cached batches for fallback', 'scenario': scenario_name}
        
        # Create loss function
        if self.flatness_loss_type == 'final':
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states,
                full_model=self.model,
                model_input_batches=cached_batches,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device=self.device,
                max_batches_for_final_loss=getattr(self, 'final_loss_max_batches', 1)
            )
        else:
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states
            )
        
        # Create dataloader
        expert_dataloader = self._create_expert_dataloader(
            hidden_states,
            batch_size=self.flatness_batch_size
        )
        
        # Use unified flatness computation method
        flatness_results = self._compute_flatness_metrics(
            expert=expert,
            loss_fn=loss_fn,
            expert_dataloader=expert_dataloader,
            flatness_method=flatness_method,
            scenario_name=scenario_name,
            trainable_only=False,
            use_landscape_fallback=use_landscape_fallback
        )
        
        # Merge results
        expert_results.update(flatness_results)
        
        # Cleanup
        del expert_dataloader
        self._clear_cuda_cache()
        
        return expert_results
    
    def _evaluate_expert_flatness_scenario_trainable_only(
        self,
        expert: nn.Module,
        expert_idx: int,
        layer_idx: int,
        hidden_states: torch.Tensor,
        cached_batches: list,
        scenario_name: str,
        scenario_description: str
    ) -> dict:
        """
        Evaluate flatness for a single expert's trainable parameters only.
        
        This is used for old experts where we only want to measure flatness
        of the trainable parts (not frozen parts).
        
        Args:
            expert: The expert module to evaluate
            expert_idx: Expert index
            layer_idx: Layer index
            hidden_states: Hidden states tensor for this expert
            cached_batches: Cached input batches (for final loss)
            scenario_name: Name of the scenario
            scenario_description: Description of the scenario
            
        Returns:
            Dictionary with flatness metrics (only for trainable parameters)
        """
        key = f"L{layer_idx}_E{expert_idx}"
        routing_mode = self.metric_routing_mode
        flatness_method = self.flatness_method
        
        print_rank_0(f"   [{scenario_name}] {key}: {scenario_description} (trainable only)", 0)
        
        # Validate hidden states
        actual_hidden_states_len = len(hidden_states) if hidden_states is not None else 0
        if actual_hidden_states_len == 0:
            print_rank_0(f"   [{scenario_name}] [Error] No tokens available, skipping", 0)
            return {'error': 'No tokens available', 'scenario': scenario_name}
        
        num_tokens = actual_hidden_states_len
        
        # Limit samples to max_samples if needed
        if num_tokens > self.flatness_max_samples:
            hidden_states = hidden_states[:self.flatness_max_samples]
            num_tokens = self.flatness_max_samples
        
        # Identify trainable parameters
        trainable_params = {name: param for name, param in expert.named_parameters() if param.requires_grad}
        
        if len(trainable_params) == 0:
            print_rank_0(f"   [{scenario_name}] [Warning] No trainable parameters found, skipping", 0)
            return {'error': 'No trainable parameters', 'scenario': scenario_name}
        
        print_rank_0(f"   [{scenario_name}] Trainable parameters: {list(trainable_params.keys())}", 0)
        
        expert_results = {
            'layer': layer_idx,
            'expert': expert_idx,
            'scenario': scenario_name,
            'scenario_description': scenario_description,
            'routing_mode': routing_mode,
            'flatness_method': flatness_method,
            'loss_type': self.flatness_loss_type,
            'num_samples': num_tokens,
            'trainable_only': True,
            'trainable_param_names': list(trainable_params.keys())
        }
        
        # Create loss function
        if self.flatness_loss_type == 'final':
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states,
                full_model=self.model,
                model_input_batches=cached_batches,
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                device=self.device,
                max_batches_for_final_loss=getattr(self, 'final_loss_max_batches', 1)
            )
        else:
            loss_fn = create_expert_loss_fn(
                loss_type=self.flatness_loss_type,
                target_hidden_states=hidden_states
            )
        
        # Create dataloader
        expert_dataloader = self._create_expert_dataloader(
            hidden_states,
            batch_size=self.flatness_batch_size
        )
        
        # Use unified flatness computation method with trainable_only=True
        flatness_results = self._compute_flatness_metrics(
            expert=expert,
            loss_fn=loss_fn,
            expert_dataloader=expert_dataloader,
            flatness_method=flatness_method,
            scenario_name=scenario_name,
            trainable_only=True,
            trainable_param_names=list(trainable_params.keys()),
            use_landscape_fallback=False  # Trainable-only doesn't use fallback
        )
        
        # Merge results
        expert_results.update(flatness_results)
        
        # Cleanup
        del expert_dataloader
        self._clear_cuda_cache()
        
        return expert_results
    
    def _get_routed_tokens(
        self, 
        hidden_states: torch.Tensor, 
        router: 'Router',
        expert_idx: int,
        min_tokens: int = 32
    ) -> torch.Tensor:
        """
        Get tokens that would be routed to a specific expert based on top-k routing.
        
        Args:
            hidden_states: Tensor of shape (num_tokens, hidden_dim)
            router: The Router module
            expert_idx: Index of the expert to get tokens for
            min_tokens: Minimum number of tokens to return (pad with random if needed)
            
        Returns:
            Tensor of routed tokens (num_routed_tokens, hidden_dim)
        """
        with torch.no_grad():
            # Reshape for router (expects batch_size, seq_len, hidden_dim)
            # We treat all tokens as a single sequence
            num_tokens, hidden_dim = hidden_states.shape
            x = hidden_states.unsqueeze(0)  # (1, num_tokens, hidden_dim)
            
            # Get routing decisions
            routing_weights, selected_experts = router(x)
            # routing_weights: (num_tokens, top_k)
            # selected_experts: (num_tokens, top_k)
            
            # Find tokens routed to this expert (in any of the top-k slots)
            expert_mask = (selected_experts == expert_idx).any(dim=-1)  # (num_tokens,)
            routed_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            if len(routed_indices) == 0:
                # No tokens routed to this expert - return random subset
                print_rank_0(f"   [Warning] No tokens routed to expert {expert_idx}, using random subset", 0)
                random_indices = torch.randperm(num_tokens, device=hidden_states.device)[:min_tokens]
                return hidden_states[random_indices], routing_weights, selected_experts
            
            routed_tokens = hidden_states[routed_indices]
            
            # Ensure minimum number of tokens for stable CKA computation
            if len(routed_indices) < min_tokens:
                # Repeat tokens to reach minimum
                repeat_factor = (min_tokens // len(routed_indices)) + 1
                routed_tokens = routed_tokens.repeat(repeat_factor, 1)[:min_tokens]
            
            return routed_tokens, routing_weights, selected_experts
    
    def _get_all_experts_routed_tokens(
        self, 
        hidden_states: torch.Tensor, 
        router: 'Router',
        expert_range: range,
        min_tokens: int = 32
    ) -> dict:
        """
        Get routed tokens for all experts in a range.
        
        Args:
            hidden_states: Tensor of shape (num_tokens, hidden_dim)
            router: The Router module
            expert_range: Range of expert indices to process
            min_tokens: Minimum tokens per expert
            
        Returns:
            Dict mapping expert_idx -> (routed_tokens, routing_stats)
        """
        result = {}
        
        with torch.no_grad():
            num_tokens, hidden_dim = hidden_states.shape
            x = hidden_states.unsqueeze(0)  # (1, num_tokens, hidden_dim)
            
            # Get routing decisions once
            routing_weights, selected_experts = router(x)
            # routing_weights: (num_tokens, top_k)
            # selected_experts: (num_tokens, top_k)
            
            for expert_idx in expert_range:
                # Find tokens routed to this expert
                expert_mask = (selected_experts == expert_idx).any(dim=-1)  # (num_tokens,)
                routed_indices = expert_mask.nonzero(as_tuple=True)[0]
                num_routed = len(routed_indices)
                
                if num_routed == 0:
                    # No tokens routed - mark for fallback evaluation
                    # Don't return random tokens, let the evaluation method decide
                    # Landscape method will be used as fallback if enabled
                    routing_stats = {
                        'num_routed': 0,  # Original routed tokens (0)
                        'num_actual': 0,  # No actual tokens (will use fallback)
                        'percentage': 0.0,
                        'fallback': 'landscape',  # Will use landscape method
                        'skip_reason': 'no_routed_tokens'
                    }
                    # Return None for tokens to indicate fallback needed
                    result[expert_idx] = (None, routing_stats)
                    continue
                else:
                    routed_tokens = hidden_states[routed_indices]
                    
                    # Ensure minimum number of tokens
                    if num_routed < min_tokens:
                        repeat_factor = (min_tokens // num_routed) + 1
                        routed_tokens = routed_tokens.repeat(repeat_factor, 1)[:min_tokens]
                    
                    # Get routing weights for these tokens
                    token_weights = routing_weights[routed_indices]
                    expert_weights = token_weights[selected_experts[routed_indices] == expert_idx]
                    
                    actual_tokens_used = len(routed_tokens)  # May be repeated if num_routed < min_tokens
                    routing_stats = {
                        'num_routed': num_routed,  # Original routed tokens
                        'num_actual': actual_tokens_used,  # Actual tokens used (may include repeats)
                        'percentage': 100.0 * num_routed / num_tokens,
                        'avg_weight': expert_weights.mean().item() if len(expert_weights) > 0 else 0.0,
                        'fallback': None
                    }
                
                result[expert_idx] = (routed_tokens, routing_stats)
        
        return result
    
    def _save_metric_results(self, i_task: int, checkpoint_name: str, results: dict):
        """
        Save metric results to JSON file.
        
        Args:
            i_task: Task ID
            checkpoint_name: Checkpoint name
            results: Results dictionary
        """
        if self.args.output_dir is None:
            return
        
        metrics_dir = os.path.join(self.args.output_dir, 'expert_metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        filename = f"metrics_task{i_task}_{checkpoint_name}.json"
        filepath = os.path.join(metrics_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print_rank_0(f"[Metrics] Results saved to {filepath}", 0)
        
        # Generate flatness visualizations after saving (for final checkpoint)
        if checkpoint_name == "after_train" and self.flatness_monitor is not None:
            try:
                print_rank_0("[Metrics] Generating flatness visualizations...", 0)
                self.flatness_monitor.plot_trends()
                self.flatness_monitor.plot_comparison()
                self.flatness_monitor.save_summary()
                print_rank_0("[Metrics] Flatness visualizations generated", 0)
            except Exception as e:
                print_rank_0(f"[Warning] Failed to generate flatness visualizations: {e}", 0)
                import traceback
                traceback.print_exc()
    
    def get_all_metric_results(self) -> dict:
        """
        Get all collected metric results.
        
        Returns:
            Dictionary of all metric results
        """
        return self.metric_results


    def dist_results_gather(self, tensor, pad_value):
        """Gather distributed results across all processes."""
        import torch.distributed as dist
        
        world_size = dist.get_world_size()
        local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        
        max_size = max(s.item() for s in all_sizes)
        
        if tensor.shape[0] < max_size:
            padding = torch.full(
                (max_size - tensor.shape[0], tensor.shape[1]),
                pad_value, dtype=tensor.dtype, device=tensor.device
            )
            tensor = torch.cat([tensor, padding], dim=0)
        
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        
        result = []
        for i, t in enumerate(gathered):
            result.append(t[:all_sizes[i].item()])
        
        return torch.cat(result, dim=0), max_size

    def evaluate(self, round, infer_task_id, task):
        """Evaluate on a specific task."""
        self.evaluate_one_task(round, infer_task_id, task)
        
    def evaluate_one_task(self, round, infer_task_id, task):
        """Evaluate on one task and save results."""
        infer_dataloader = self.test_task_list[task]
        progress_bar = tqdm(total=len(infer_dataloader), leave=True, disable=(self.args.global_rank != 0))
        
        def prediction(model, infer_dataloader):
            predicted_sequences = []
            sources_sequences = []
            label_sequences = []
            model.eval()

            for step, batch in enumerate(infer_dataloader):
                ground_truths_ids = self.tokenizer(
                    batch['gts'], 
                    truncation=True,
                    max_length=self.args.max_ans_len,
                    add_special_tokens=False,
                    padding='max_length',
                    return_tensors='pt'
                )['input_ids'].to(self.device)
                del batch['gts']
                del batch['sources']
                batch = to_device(batch, self.device)
                
                if self.args.global_rank == 0:
                    progress_bar.update(1)

                with torch.no_grad():
                    generate_ids = model.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_new_tokens=self.args.max_ans_len,
                        bos_token_id=self.tokenizer.bos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.unk_token_id,
                        generation_config=generation_config,
                        use_cache=False
                    )

                gathered_ids, max_seq_len = self.dist_results_gather(generate_ids, self.tokenizer.eos_token_id)
                gathered_labels, max_label_len = self.dist_results_gather(ground_truths_ids, self.tokenizer.eos_token_id)

                if self.args.global_rank <= 0:
                    input_len = batch['input_ids'].shape[1]
                    sou_sequences = self.tokenizer.batch_decode(
                        gathered_ids[:, :input_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    pre_sequences = self.tokenizer.batch_decode(
                        gathered_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    lab_sequences = self.tokenizer.batch_decode(
                        gathered_labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    
                    predicted_sequences.extend(pre_sequences)
                    sources_sequences.extend(sou_sequences)
                    label_sequences.extend(lab_sequences)

            return sources_sequences, predicted_sequences, label_sequences

        def save_inference_results(evaluation_result, sources_sequences, predicted_sequences,
                                   ground_truths, round, i_task, task):
            df = {
                "eval": evaluation_result, 
                'prompts': sources_sequences, 
                'results': predicted_sequences,
                'labels': ground_truths
            }
            if not os.path.exists(self.args.output_dir):
                os.makedirs(self.args.output_dir)

            with open(os.path.join(self.args.output_dir, f"results-{round}-{i_task}-{task}.json"), "w+", encoding='utf-8') as file:
                json.dump(df, file, ensure_ascii=False, indent=4)

        print_rank_0("***** Start inference *****", self.args.global_rank)
        sources_sequences, predicted_sequences, ground_truths = prediction(self.model, infer_dataloader)

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


class ExpertLossWrapper(nn.Module):
    """
    Wrapper to create a loss function from expert output for Hessian computation.
    """
    def __init__(self, expert: nn.Module, device: torch.device):
        super().__init__()
        self.expert = expert
        self.device = device
    
    def forward(self, x):
        """Compute expert output and return a scalar loss."""
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device)
        output = self.expert(x)
        return (output ** 2).mean()

