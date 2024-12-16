import transformers
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
import lightning as pl
import torch.nn as nn
import math
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import torch
import torch.nn.functional as F
import inspect

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaMLP,LlamaRMSNorm,LlamaRotaryEmbedding, LlamaModel,LlamaForCausalLM
from transformers import LlamaModel
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
# from peft.utils import get_peft_model
from transformers.models.llama.configuration_llama import LlamaConfig
from functools import partial
from torch.utils.checkpoint import checkpoint
model_id = "meta-llama/Llama-3.2-1B"

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# print(model)
# lconfig = LoraConfig()



class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaSdpaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class LlamaCausalLM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)
         # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, idx,targets=None,past_key_values=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.intermediate_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.intermediate_size}"
        position_ids = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the Llama model itself
        # token embeddings of shape (b, t, n_embd)
        inputs_embeds = self.embed_tokens(idx)
        
        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)
        # position embeddings of shape (t, n_embd)
        hidden_states =inputs_embeds
        for decoder_layer in self.layers:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

            gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids.unsqueeze(0),
                    past_key_value=past_key_values,
                    output_attentions=False,
                )
            
            hidden_states = layer_outputs[0]
            if self.config.use_cache:
                next_decoder_cache = None

        hidden_states = self.norm(hidden_states)

        next_cache = next_decoder_cache
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(hidden_states)
            
            # logits = BaseModelOutputWithPast(logits,next_cache,None,None)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(hidden_states[:, [-1], :])
            
            # logits = BaseModelOutputWithPast(logits,next_cache,None,None)
            loss = None

        return logits, loss

    
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.intermediate_size
                else idx[:, -self.config.intermediate_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def from_pretrained(self,id):
        print("loading weights from pretrained gpt: %s" % id)
        model = LlamaCausalLM(self.config)
        hf = LlamaModel(self.config).from_pretrained(id)
        # model_hf = hf.model 
        # lm_hf = hf.lm_head
        # sd_lm_hf = lm_hf #The retards at META decided to create another additional class for the MLP head. Doing this to combine both of them
        
        # sd_lm_hf_keys = sd_lm_hf.keys()
        # del hf
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_hf = hf.state_dict()
        # sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        

        # Iterate through both lists simultaneously
      
        # sd_keys_hf = [
        #     k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        # ]  # ignore these, just a buffer
        # sd_keys_hf = [
        #     k for k in sd_keys_hf if not k.endswith(".attn.bias")
        # ]  # same, just the mask (buffer)
        # assert len(sd_keys_hf) == len(
        #     sd_keys
        # ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            
            # vanilla copy over the other parameters
            
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
        # assert lm_hf.weight.shape == sd['lm_head.weight'].shape
        # sd['lm_head.weight'].copy_(lm_hf.weight)
        # del lm_hf,sd_hf,sd_keys_hf
        print("Loading done")
        
        return model

model = LlamaCausalLM(config).from_pretrained(model_id)


class LlamaChessLightning(pl.LightningModule):
    def __init__(self, config,rank):
        super().__init__()
        self.config = config
        self.model = LlamaCausalLM(config).from_pretrained(model_id) #LlamaForCausalLM(self.config).from_pretrained(model_id,quantization_config=bnb_config) #
        # print(self.model.state_dict().keys())
        lora_config = LoraConfig(
                                r=rank,
                                lora_alpha=rank,
                                init_lora_weights="gaussian",
                                target_modules=["q_proj","k_proj","v_proj","o_proj"],
                            )
        self.model = prepare_model_for_kbit_training(self.model)
        self.lora_model = get_peft_model(self.model,lora_config)
        print("Reducing number of parameters using LoRA to: %.2fM" % (self.get_num_params() / 1e6,))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters(
            {
                "vocab_size": 128256,
            }
        )
        # print(self.lora_model.state_dict().keys())

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad is True)
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def get_model_config(self):
        return self.config
    
    def forward(self, x, y):
        # print(x.shape,y.shape)
        return self.model(x, labels=y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(x.shape,y.shape)
        _, loss = self(x, y)

        self.log(
            "train_loss",
            loss if loss else -1.0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            "val_loss",
            loss if loss else -1.0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss


    def configure_optimizers(self):
        # start with all of the candidate parameters
        weight_decay=0.0
        device_type = "cuda" if self.on_gpu else "cpu"
        learning_rate=1e-3
        betas=(0.9,0.95)#(self.hparams["beta1"], self.hparams["beta2"])

        param_dict = {pn: p for pn, p in self.lora_model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=4000
        )

        return [optimizer], [scheduler]
    
    
