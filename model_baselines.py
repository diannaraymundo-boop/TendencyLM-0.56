"""对照模型:GRU 和小 Transformer。

没有对照就无法判断设计有没有用。所有模型都有 heartbeat 接口,由 build_model 创建。
对照模型自己训练时用整段窗口,比较时都按“一次心跳预测一个词”算。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model_flow import FlowModel
from tokenizer_tool import BOS_ID


def build_model(cfg: ModelConfig):
    if cfg.model_type == "flow":
        return FlowModel(cfg)
    if cfg.model_type == "gru":
        return GRUBaseline(cfg)
    if cfg.model_type == "transformer":
        return TransformerBaseline(cfg)
    raise ValueError(f"未知的模型类型:{cfg.model_type}")


class BaselineModel(nn.Module):
    def heartbeat(self, tokens, external, mask, beats):
        """只取最后一个位置的预测。对照模型没有时钟和倾向,长度不同的样本分开算。"""
        lengths = mask.sum(1)
        logits = torch.zeros(tokens.size(0), self.cfg.vocab_size, device=tokens.device)
        for length in lengths.unique().tolist():
            chosen = lengths == length
            output, _ = self(tokens[chosen, :length], external[chosen, :length])
            logits[chosen] = output[:, -1]
        return logits


# =====================================================================
# GRU:不回头读历史,状态是一个向量
# =====================================================================


class GRUBaseline(BaselineModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.source_embed = nn.Embedding(2, cfg.embed_dim)
        self.cell = nn.GRUCell(cfg.embed_dim, cfg.gru_hidden)
        self.out_embed = nn.Linear(cfg.gru_hidden, cfg.embed_dim, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.source_embed.weight, std=0.02)

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.gru_hidden, device=device)

    @staticmethod
    def detach_state(state):
        return state.detach()

    def forward(self, tokens, is_external, state=None):
        if state is None:
            state = self.init_state(tokens.size(0), tokens.device)
        inputs = self.token_embed(tokens) + self.source_embed(is_external.long())
        hidden = []
        for i in range(tokens.size(1)):
            state = state * (tokens[:, i] != BOS_ID).float().unsqueeze(1)
            state = self.cell(inputs[:, i], state)
            hidden.append(state)
        output = self.out_embed(torch.stack(hidden, 1))
        return output @ self.token_embed.weight.t() + self.out_bias, state


# =====================================================================
# Transformer:每步都能回看前面的 token,不维护状态
# =====================================================================


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.query_key_value = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, mask):
        batch, length, dim = x.shape
        query, key, value = self.query_key_value(self.norm1(x)).chunk(3, -1)
        split = lambda t: t.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(split(query), split(key), split(value), attn_mask=mask)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, length, dim))
        return x + self.feed_forward(self.norm2(x))


class TransformerBaseline(BaselineModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.tf_dim)
        self.source_embed = nn.Embedding(2, cfg.tf_dim)
        self.position_embed = nn.Embedding(cfg.tf_context, cfg.tf_dim)
        self.blocks = nn.ModuleList(TransformerBlock(cfg.tf_dim, cfg.tf_heads) for _ in range(cfg.tf_layers))
        self.norm = nn.LayerNorm(cfg.tf_dim)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.source_embed.weight, std=0.02)
        nn.init.normal_(self.position_embed.weight, std=0.02)

    def init_state(self, batch_size, device):
        empty = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        return empty, empty.bool()

    @staticmethod
    def detach_state(state):
        return state

    def forward(self, tokens, is_external, state=None):
        if state is None:
            state = self.init_state(tokens.size(0), tokens.device)
        length = tokens.size(1)
        all_tokens = torch.cat([state[0], tokens], 1)[:, -self.cfg.tf_context :]
        all_external = torch.cat([state[1], is_external], 1)[:, -self.cfg.tf_context :]
        total = all_tokens.size(1)

        # 只能看同一段对话里自己之前的 token
        conversation = (all_tokens == BOS_ID).cumsum(1)
        same = conversation.unsqueeze(1) == conversation.unsqueeze(2)
        causal = torch.ones(total, total, dtype=torch.bool, device=tokens.device).tril()
        mask = (same & causal).unsqueeze(1)

        # 位置从这段对话的开头数起
        index = torch.arange(total, device=tokens.device).expand_as(all_tokens)
        last_start = torch.where(all_tokens == BOS_ID, index, torch.zeros_like(index)).cummax(1).values
        x = self.token_embed(all_tokens) + self.source_embed(all_external.long()) + self.position_embed(index - last_start)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.norm(x[:, -length:]) @ self.token_embed.weight.t() + self.out_bias

        # 留下一部分 token 给下一个窗口当上下文
        keep = max(0, self.cfg.tf_context - length)
        new_state = (all_tokens[:, -keep:] if keep else all_tokens[:, :0], all_external[:, -keep:] if keep else all_external[:, :0])
        return logits, new_state
