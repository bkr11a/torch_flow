"""
One Cycle Learning Rate scheduler for PyTorch.

Implementation of Leslie Smith's One Cycle LR policy.
"""

__author__ = "Brad Rice"
__version__ = "1.0.0"

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class OneCycleLR(_LRScheduler):
    \"\"\"
    Implements the One Cycle policy (Leslie Smith, 2018).
    
    Combines a single learning rate cycle with momentum cycling for 
    single-pass training.
    \"\"\"
    
    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        \"\"\"
        Initialize One Cycle scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            max_lr: Maximum learning rate
            total_steps: Total number of steps in training
            pct_start: Percentage of cycle in increasing phase
            anneal_strategy: 'cos' or 'linear' for annealing
            cycle_momentum: Whether to cycle momentum
            base_momentum: Base (maximum) momentum
            max_momentum: Maximum momentum (minimum when cycling)
            div_factor: Initial lr = max_lr / div_factor
            final_div_factor: Final lr = initial_lr / final_div_factor
            last_epoch: Last epoch (for resuming)
            verbose: Print learning rate changes
        \"\"\"
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Validate
        if div_factor <= 0:
            raise ValueError(\"div_factor must be > 0\")
        if final_div_factor <= 0:
            raise ValueError(\"final_div_factor must be > 0\")
        
        # Calculate initial and final learning rates
        self.initial_lr = self.max_lr / self.div_factor
        self.final_lr = self.initial_lr / self.final_div_factor
        
        # Set initial learning rate
        for group in self.optimizer.param_groups:
            group['initial_lr'] = self.initial_lr
            group['max_lr'] = self.max_lr
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        \"\"\"Calculate learning rate for current step.\"\"\"
        cycle_size = int(self.total_steps * self.pct_start)
        current_step = self.last_epoch
        
        if current_step < cycle_size:
            # Increasing phase
            progress = current_step / cycle_size
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                    (1 - math.cos(math.pi * progress)) / 2
                )
            else:  # linear
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (current_step - cycle_size) / (self.total_steps - cycle_size)
            progress = min(progress, 1.0)
            if self.anneal_strategy == 'cos':
                lr = self.max_lr + (self.final_lr - self.max_lr) * (
                    (1 + math.cos(math.pi * progress)) / 2
                )
            else:  # linear
                lr = self.max_lr + (self.final_lr - self.max_lr) * progress
        
        return [lr for _ in self.optimizer.param_groups]
    
    def get_momentum(self):
        \"\"\"Calculate momentum for current step.\"\"\"
        if not self.cycle_momentum:
            return [group.get('momentum', 0) for group in self.optimizer.param_groups]
        
        cycle_size = int(self.total_steps * self.pct_start)
        current_step = self.last_epoch
        
        if current_step < cycle_size:
            # Decreasing momentum during increasing LR
            progress = current_step / cycle_size
            momentum = self.max_momentum - (self.max_momentum - self.base_momentum) * progress
        else:
            # Increasing momentum during decreasing LR
            progress = (current_step - cycle_size) / (self.total_steps - cycle_size)
            progress = min(progress, 1.0)
            momentum = self.base_momentum + (self.max_momentum - self.base_momentum) * progress
        
        return [momentum for _ in self.optimizer.param_groups]
    
    def step(self, epoch: int = None):
        \"\"\"Update learning rate and momentum.\"\"\"
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # Update learning rates
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        
        # Update momentum if supported
        if self.cycle_momentum:
            momentums = self.get_momentum()
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if 'momentum' in param_group:
                    param_group['momentum'] = momentum
        
        if self.verbose:
            print(f\"Epoch {epoch}: LR={lrs[0]:.2e}\")
