from dataclasses import dataclass, field

import draccus


@dataclass
class DataConfig:
    num_workers: int = 4
    root: str = "data/wiki"


@dataclass
class OptimConfig:
    lr: float = 3e-4
    warmup_steps: int = 1000
    total_steps: int = 50000

    weight_decay: float = 0.05
    grad_norm_clip: float = 1.0
    logit_reg_weight: float = 1e-4

    batch_size: int = 32
    grad_accum_steps: int = 1

    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8


@dataclass
class ModelConfig(draccus.PluginRegistry, discover_packages_path="gateloop.models"):
    seq_len: int = 512
    vocab_size: int = -1  # -1 means automatically determined from tokenizer

    @property
    def model_class(self):
        raise NotImplementedError()

    def build(self, key=None):
        return self.model_class(self, key=key)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    out_dir: str = "./_train_out"
    checkpoint_period: int = 2000
    log_period: int = 10
    val_period: int = 200
    val_examples: int = 4096
    random_seed: int = 1337
