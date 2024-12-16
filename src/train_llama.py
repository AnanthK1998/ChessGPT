

from dataclasses import asdict

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import DataConfig, ModelConfig, TrainConfig
from src.datamodules import GPTChessDataModule
from src.llama import LlamaChessLightning
from transformers.models.llama.configuration_llama import LlamaConfig

def main():
    rope_scaling = {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  }
    model_config = LlamaConfig(
                _attn_implementation_autoset=True,
                hidden_size=2048,
                intermediate_size=8192,
                num_attention_heads=32,
                num_hidden_layers=16,
                rms_norm_eps=1e-05,
                num_key_value_heads=8,
                vocab_size=128256,
                rope_theta=500000.0,
                rope_scaling=rope_scaling,
                max_position_embeddings=131072,
                bos_token_id=128000,
                eos_token_id=128001,
                torch_dtype=torch.bfloat16,
                tie_word_embeddings=True,
                use_cache=True,
                learning_rate=1e-3,
                decay_iters=4000,
                beta1=0.9,
                beta2=0.95,
                weight_decay=0.1
                
            )

    data_config = DataConfig(
        dataset_path="data/stockfish",
        file_path="stockfish_dataset_blocks.zip",
        batch_size=4,
        num_workers=24,
    )

    train_config = TrainConfig(
        max_epochs=50,
        val_check_interval=0.01,
        log_every_n_steps=10,
        overfit_batches=0,
        checkpoint_path="checkpoints/",
        checkpoint_interval=10000,
        wandb_project="chessgpt",
        wandb_tags=["runpod_stockfish_run"],
        gradient_accumulation_steps=10,
    )
    torch.manual_seed(1337)
    # torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_float32_matmul_precision("high")

    model = LlamaChessLightning(model_config,8)
    dm = GPTChessDataModule(
        **asdict(data_config), block_size=model_config.intermediate_size
    )

    ckpt_callback = ModelCheckpoint(
        save_last=True,
        dirpath=train_config.checkpoint_path,
        every_n_train_steps=train_config.checkpoint_interval,
        verbose=True,
    )

    wandb_logger = WandbLogger(
        project=train_config.wandb_project,
        tags=train_config.wandb_tags,
        # resume="must",
        # id="exfy7bt4",
    )

    trainer = pl.Trainer(
        max_epochs=train_config.max_epochs,
        log_every_n_steps=train_config.log_every_n_steps,
        val_check_interval=train_config.val_check_interval,
        overfit_batches=train_config.overfit_batches,
        accumulate_grad_batches=train_config.gradient_accumulation_steps,
        logger=wandb_logger,
        callbacks=[ckpt_callback],
        precision="bf16-mixed",
    )
    trainer.fit(
        model,
        datamodule=dm,
        # ckpt_path="/opt/joao/repos/xp/chessgpt/checkpoints/last-v4.ckpt",
    )


if __name__ == "__main__":
    main()
