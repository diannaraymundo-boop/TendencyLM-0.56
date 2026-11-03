"""意识流模型(第五版)。

用户定的链条:
    1. 注意力一次看完全文,给每个输入词叠上倾向。修正后的词 = 静态意思 + 倾向,
       表示这个词在这段话里的抽象含义,不再严格对应词表。
    2. 推理读修正后的词,产出意识流。意识流本身带着上下文,所以说话时不再重复做注意力修正。
    3. 语言组件把意识流翻译成严格对照词表的字。
    4. 注意力检查翻译出的字:给每个字一个置信度,并给全文(输入 + 输出)重新覆盖一次倾向。
    5. 推理读覆盖后的全文,把置信度低的字当成空重新填,置信度高的保留。前后的字都看得见。
    6. 再翻译,再检查,直到没有低置信度的字或到上限。

倾向每次都是覆盖:新的倾向叠在静态意思上,替换旧的,不往上累加。

说话时的一次心跳:
    1. 当前时间   固定初始时间 + 时钟偏移(第几拍)
    2. 读         每列各自过同一个推理网络(已算好)。读原文(输入词)和读自己说过的列分开,
                  各分几路,每路按一张比例表把列合起来,比例由当前时间和每列的键决定。
                  键在输入时(或这一列加进来时)就算好了,这里只换查询,不重算倾向
    3. 推理       读到的合起来,产出这一拍的意识流
    4. 翻译       意识流 → 字
    5. 新的一列 = 意识流原样 + 说出口的那个字(静态意思过同一个推理网络),进入下一拍。
       训练时说出口的字用答案的字,生成时用自己说的字;说错了由检查和填空来改。
       2026-11-03 第一版只接意识流、听不到自己说了什么,真实文本上学不会字和字怎么接

每拍的当前时间、读的查询只由第几拍和站的位置决定,和说了什么无关,开始说话前一次算好(plan),
每拍直接取用。结果和每拍现算一样,只是每拍少发几十个小任务,见 README“提速”。

意识流起点(use_seed):输入末尾放一个空位,和词一起被注意力修正,修正后就是注意力对全文的理解,
直接成为意识流的第一列,不和输入词一起按比例平均。检查时也放一个,填空时读。

“改倾向的权重表”(用户定的:注意力认为每个词该改多少)只在输入时和检查时用。
2026-11-03 以前,说话时合并也用同一张表,原文和说过的列挤在一起分,排查发现语序在这一步被平均掉、
第 2 拍以后几乎只听自己,所以拆开。旧做法留在 split_reading = False。
"""
import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# =====================================================================
# 时钟和顺序
# =====================================================================


class Clock(nn.Module):
    """把“走了多少格”翻译成向量。0 格时偏移正好是 0。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.max_beats = cfg.max_outputs
        self.log_period = nn.Parameter(torch.linspace(math.log(2.0), math.log(cfg.max_outputs * 4.0), cfg.flow_dials))
        self.net = nn.Sequential(
            nn.Linear(cfg.flow_dials * 2 + 1, cfg.flow_time_dim),
            nn.GELU(),
            nn.Linear(cfg.flow_time_dim, cfg.flow_time_dim, bias=False),
        )

    def _translate(self, beats):
        beats = beats.float().unsqueeze(-1)
        angle = 2 * math.pi * beats / self.log_period.exp()
        elapsed = torch.log1p(beats) / math.log1p(self.max_beats)
        return self.net(torch.cat([angle.sin(), angle.cos(), elapsed], -1))

    def forward(self, beats):
        return self._translate(beats) - self._translate(torch.zeros_like(beats))


class OrderRotation(nn.Module):
    """同一个词在不同位置的表征不同,顺序靠训练内化。"""

    def __init__(self, dim):
        super().__init__()
        periods = torch.logspace(1.0, 12.0, dim // 2, base=2.0)
        self.register_buffer("speed", 2 * math.pi / periods, persistent=False)

    def forward(self, x, order):
        """x: (批, ..., 列, 维度),order: (批, 列)。"""
        angle = order.unsqueeze(-1).float() * self.speed
        angle = angle.view(order.size(0), *[1] * (x.dim() - 3), *angle.shape[1:])
        first, second = x[..., 0::2], x[..., 1::2]
        return torch.stack([first * angle.cos() - second * angle.sin(), first * angle.sin() + second * angle.cos()], -1).flatten(-2)


def positions_to_order(mask):
    """每个真实 token 离“最后一个真实 token”有多远。填充位置是 0。"""
    total = mask.sum(1, keepdim=True)
    rank = mask.long().cumsum(1) - 1
    return torch.where(mask, total - 1 - rank, torch.zeros_like(rank)).float()


def absolute_positions(mask):
    """每个真实 token 是第几个(从 0 数),中间夹着填充也照样连着数。"""
    return (mask.long().cumsum(1) - 1).clamp_min(0)


class AttentionRound(nn.Module):
    """一轮 token 互相看。当前时间直接改变每个 token 想看谁。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.flow_dim
        self.heads = cfg.flow_heads
        self.norm1 = nn.LayerNorm(D)
        self.query_key_value = nn.Linear(D, D * 3, bias=False)
        self.time_to_query = nn.Linear(cfg.flow_time_dim, D * 2)
        self.rotation = OrderRotation(D // cfg.flow_heads) if cfg.use_order else None
        self.out = nn.Linear(D, D, bias=False)
        self.norm2 = nn.LayerNorm(D)
        self.feed_forward = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))

    def forward(self, x, mask, order, now):
        batch, columns, dim = x.shape
        query, key, value = self.query_key_value(self.norm1(x)).chunk(3, -1)
        scale, shift = self.time_to_query(now).unsqueeze(1).chunk(2, -1)
        query = query * (1 + scale) + shift
        split = lambda t: t.view(batch, columns, self.heads, dim // self.heads).transpose(1, 2)
        query, key, value = split(query), split(key), split(value)
        if self.rotation is not None:
            query, key = self.rotation(query, order), self.rotation(key, order)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None, None, :])
        x = x + self.out(attended.transpose(1, 2).reshape(batch, columns, dim))
        return x + self.feed_forward(self.norm2(x))


# =====================================================================
# 注意力:输入时、检查时各完整跑一次
# =====================================================================


class TendencyAttention(nn.Module):
    """完整注意力:给每个词的改动(倾向的方向)和权重表的键。

    权重表 = 键和查询的夹角。查询完全来自当前时间,按“现在站在哪个位置”转一个角度;
    键在完整跑完时按自己的位置转好,之后每一拍只要重新转查询、做一次点乘,不用重跑注意力。
    这和“每个键按离最新一个词多远来转”完全等价(转角度只改方向,两边一起转夹角不变)。
    """

    def __init__(self, cfg: ModelConfig, rounds: int):
        super().__init__()
        D = cfg.flow_dim
        self.rounds = nn.ModuleList(AttentionRound(cfg) for _ in range(rounds))
        self.norm = nn.LayerNorm(D)
        self.weight_key = nn.Linear(D, D, bias=False)
        self.time_to_weight = nn.Linear(cfg.flow_time_dim, D)
        self.rotation = OrderRotation(D) if cfg.use_order else None
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(10.0)))
        self.change_head = nn.Linear(D, D)
        nn.init.normal_(self.change_head.weight, std=0.02)
        nn.init.zeros_(self.change_head.bias)
        # 说话和填空时读每个词用的键,和权重表的键分开
        self.read_key = nn.Linear(D, D, bias=False) if cfg.split_reading else None
        # 每个词的改动幅度各自定(0~1,词和词不互相抢)时的偏置
        self.amount_bias = nn.Parameter(torch.zeros(())) if cfg.independent_amount else None

    def forward(self, meaning, mask, now):
        """返回 (特征, 改动, 键)。"""
        x = meaning
        order = positions_to_order(mask)
        for attention_round in self.rounds:
            x = attention_round(x, mask, order, now)
        feature = self.norm(x)
        return feature, self.change_head(feature), self.place(self.weight_key(feature), absolute_positions(mask))

    def place(self, key, position):
        """把键按自己的位置转好、归一化。key: (批, 列, 维度),position: (批, 列)。"""
        key = F.normalize(key, dim=-1)
        return self.rotation(key, -position) if self.rotation is not None else key

    def query(self, now, position):
        """这一拍的查询。now: (批, [空数,] 时间维度),position: (批,) 或 (批, 空数),现在站在哪个位置。"""
        query = F.normalize(self.time_to_weight(now), dim=-1)
        if self.rotation is None:
            return query
        if position.dim() == 1:
            return self.rotation(query.unsqueeze(1), -position.unsqueeze(1)).squeeze(1)
        return self.rotation(query, -position)

    def weigh(self, scores, mask):
        """夹角 → 权重表,每张表加起来是 1。"""
        return torch.softmax((scores * self.log_sharpness.exp()).masked_fill(~mask, float("-inf")), -1)

    def amount(self, scores, mask):
        """夹角 → 每个词各自改多少(0~1),词和词不互相抢。"""
        return torch.sigmoid(scores * self.log_sharpness.exp() + self.amount_bias) * mask


# =====================================================================
# 读:说话和填空时,推理按什么比例把各列合起来
# =====================================================================


class Reader(nn.Module):
    """分几路读一组列。每路一张比例表(加起来是 1),只读每列的一段(宽度 ÷ 路数),几路读到的接起来。

    几路各看各的,一拍可以同时看问句、看人名,不会被平均成一团。
    比例 = 这一路的查询和每列的键的夹角:查询来自当前时间(第几拍),按现在站的位置转角度;
    键由注意力给(输入词、检查后的全文)或由说过的那一列自己算,按那一列的位置转好。
    查询不看推理当前在想什么(用户认为推理不需要内置注意力),除非打开 read_by_thought。
    can_skip:每一路多一个“这拍不读”的选项,分数只由当前时间决定,选它的那部分读到 0。
    """

    def __init__(self, cfg: ModelConfig, can_skip=False):
        super().__init__()
        D, P = cfg.flow_dim, cfg.read_paths
        self.paths = P
        self.time_to_query = nn.Linear(cfg.flow_time_dim, D)
        self.thought_to_query = nn.Linear(cfg.flow_stream_dim, D, bias=False) if cfg.read_by_thought else None
        self.rotation = OrderRotation(D // P) if cfg.use_order else None
        self.log_sharpness = nn.Parameter(torch.full((P,), math.log(10.0)))
        self.skip = nn.Linear(cfg.flow_time_dim, P) if can_skip else None
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def _split(self, x, position):
        """(批, 个数, D) → 分路、各自归一化、按位置转好:(批, 路, 个数, D/路)。"""
        batch, count, _ = x.shape
        x = F.normalize(x.view(batch, count, self.paths, -1), dim=-1).transpose(1, 2)
        return self.rotation(x, -position) if self.rotation is not None else x

    def place(self, key, position):
        """键。key: (批, 列, D),position: (批, 列) 每列的位置。"""
        return self._split(key, position)

    def query(self, now, position, thought=None):
        """查询。now: (批, 个数, 时间维度),position: (批, 个数) 现在站在哪个位置,thought: (批, 个数, 意识流宽度)。"""
        query = self.time_to_query(now)
        if self.thought_to_query is not None and thought is not None:
            query = query + self.thought_to_query(thought)
        return self._split(query, position)

    def skip_scores(self, now):
        """“这拍不读”的分数。now: (批, 个数, 时间维度) → (批, 路, 个数, 1);没有这个选项时是 None。"""
        return None if self.skip is None else self.skip(now).transpose(1, 2).unsqueeze(-1)

    def weigh(self, query, keys, mask, skip=None):
        """比例表。query (批, 路, 个数, d),keys (批, 路, 列, d),mask (批, 列) → (批, 路, 个数, 列)。
        每路加起来是 1(有“不读”选项时,选它的那部分不算在内,加起来不到 1);一列都没有的行全是 0。"""
        scores = (query @ keys.transpose(-1, -2)) * self.log_sharpness.exp().view(1, -1, 1, 1)
        visible = mask[:, None, None, :].expand_as(scores)
        if skip is not None:
            scores = torch.cat([skip, scores], -1)
            visible = torch.cat([visible.new_ones(skip.shape), visible], -1)
        weights = torch.softmax(scores.masked_fill(~visible, -1e9), -1) * visible
        return weights if skip is None else weights[..., 1:]

    def read(self, query, keys, values, mask, skip=None):
        """values (批, 列, 宽度)。返回 (读到的 (批, 个数, 宽度),比例 (批, 路, 个数, 列))。

        每一路先按自己的比例把整列加起来,再只留自己那一段。多算的几段不费事;要是先把 values
        切成几段再读,每拍都会复制一份 values 留着倒推,真实尺寸一批 32 段要 13 GB 以上。
        """
        weights = self.weigh(query, keys, mask, skip)
        batch, paths, count, columns = weights.shape
        full = weights.transpose(1, 2).reshape(batch, count * paths, columns) @ values
        full = full.view(batch, count, paths, paths, -1)  # 第 p 路读到的第 p' 段
        read = full.diagonal(dim1=2, dim2=3).transpose(-1, -2).reshape(batch, count, -1)
        return read, weights


# =====================================================================
# 状态
# =====================================================================


class SpeakState(NamedTuple):
    """说一段话时攒下来的东西。输入部分说话时不变,每拍加一列。"""

    each: torch.Tensor          # (批, 输入列, 意识流宽度) 每个修正后的输入词过推理网络的结果
    keys: torch.Tensor          # 输入词读的时候用的键:(批, 路, 输入列, 维度/路);旧做法是权重表的键 (批, 输入列, 维度)
    mask: torch.Tensor          # (批, 输入列) 是不是真实的词
    lengths: torch.Tensor       # (批,) 输入有多少个真实的词
    beat_each: torch.Tensor     # (批, 列数, 意识流宽度) 意识流起点(如果有)+ 每拍加的一列:意识流 + 说出口的字
    beat_keys: torch.Tensor     # 这些列的键:(批, 路, 列数, 维度/路);旧做法 (批, 列数, 维度)
    start: int = 0              # beat_each 开头有几列不是说话时加的(意识流起点)


class Review(NamedTuple):
    """检查的结果。全文 = 输入 + 翻译出的字(+ 意识流起点,如果有,放在最后一列)。"""

    confidence: torch.Tensor    # (批, 字数) 每个字的置信度,0~1
    vectors: torch.Tensor       # (批, 全文列, 维度) 覆盖倾向后的全文
    keys: torch.Tensor          # 填空读全文用的键:(批, 路, 全文列, 维度/路);旧做法是权重表的键 (批, 全文列, 维度)
    mask: torch.Tensor          # (批, 全文列)
    lengths: torch.Tensor       # (批,) 输入有多少个真实的词
    weights: torch.Tensor       # (批, 输入 + 字的列) 检查时每个词的改动幅度(由权重表定)


class BeatPlan(NamedTuple):
    """说话时每拍都要用、又和说了什么无关的东西,开始说话前一次算好(见 FlowModel.plan)。

    每一项都按拍拆成一串,第 i 个是第 first + i 拍的。
    """

    first: int                                       # 从第几拍开始
    now: Tuple[torch.Tensor, ...]                    # 每拍的当前时间 (时间维度,)
    input_query: Optional[Tuple[torch.Tensor, ...]]  # 读原文的查询 (批, 路, 维度/路);旧做法、read_by_thought 时是 None,每拍现算
    self_query: Optional[Tuple[torch.Tensor, ...]]   # 读自己说过的查询,同上
    skip: Optional[Tuple[torch.Tensor, ...]]         # “这拍不读”的分数 (批, 路, 1);没有这个选项或旧做法时是 None


# =====================================================================
# 模型本体
# =====================================================================


class FlowModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.flow_dim % cfg.flow_heads or (cfg.flow_dim // cfg.flow_heads) % 2 or cfg.flow_dim % 2:
            raise ValueError("flow_dim 必须能被 flow_heads 整除,flow_dim 和每路的长度都必须是偶数")
        if cfg.split_reading:
            if cfg.flow_dim % cfg.read_paths or (cfg.flow_dim // cfg.read_paths) % 2 or cfg.flow_stream_dim % cfg.read_paths:
                raise ValueError("flow_dim、flow_stream_dim 必须能被 read_paths 整除,flow_dim 每路的长度必须是偶数")
        elif cfg.read_paths != 1 or cfg.read_by_thought or cfg.use_seed:
            raise ValueError("read_paths、read_by_thought、use_seed 只在 split_reading 打开时有用")
        self.cfg = cfg
        D, T, S = cfg.flow_dim, cfg.flow_time_dim, cfg.flow_stream_dim

        # 静态意思表。翻译时也拿它当对照表
        self.embed = nn.Embedding(cfg.vocab_size, D)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.initial_time = nn.Parameter(torch.empty(T))
        nn.init.normal_(self.initial_time, std=0.02)
        self.clock = Clock(cfg)

        self.input_attention = TendencyAttention(cfg, cfg.attn_rounds)
        self.review_attention = TendencyAttention(cfg, cfg.review_rounds)
        self.confidence_head = nn.Linear(D, 1)
        # 意识流起点:放在输入末尾和词一起被注意力修正,自己没有本意,修正后就是注意力对全文的理解
        self.seed = nn.Parameter(torch.randn(D) * 0.02) if cfg.use_seed else None

        # 推理:每列各自过 token_net,按权重合起来,再过 merge_net 得到意识流
        self.token_net = nn.Sequential(nn.Linear(D, S), nn.GELU(), nn.Linear(S, S), nn.GELU(), nn.Linear(S, S))
        self.merge_net = nn.Sequential(nn.Linear(S, S), nn.GELU(), nn.Linear(S, S))
        self.stream_norm = nn.LayerNorm(S)
        self.stream_key = nn.Linear(S, D, bias=False)
        if cfg.split_reading:
            self.read_input = Reader(cfg)   # 读原文:输入词(填空时是全文的输入部分)。每拍一定读
            self.read_self = Reader(cfg, can_skip=cfg.self_can_skip)  # 读自己说过的:每拍加的列(填空时是翻译出的字)
        # 意识流进入下一拍时的幅度。归一化后的意识流长度约 √S(默认约 22),而字和输入词各自过网络后
        # 只有约 0.6,原样相加会把它们盖住。从 1/√S 开始,让几样东西长度相当,之后由训练决定
        self.stream_gain = nn.Parameter(torch.tensor(1.0 / math.sqrt(S)))

        # 语言组件:意识流 → 字
        self.translate_net = nn.Linear(S, D)
        self.translate_norm = nn.LayerNorm(D)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    # ---------- 当前时间 ----------

    def current_time(self, beats):
        now = self.initial_time.expand(*beats.shape, -1)
        return now + self.clock(beats) if self.cfg.use_clock else now

    # ---------- 推理和翻译 ----------

    def think(self, merged):
        """合起来的结果 → 意识流。"""
        return self.stream_norm(self.merge_net(merged))

    def translate(self, stream):
        """意识流 → 每个字的分数。"""
        return self.translate_norm(self.translate_net(stream)) @ self.embed.weight.t() + self.out_bias

    def _overlay(self, meaning, change, amount):
        """倾向 = 幅度 × 改动,叠在静态意思上(覆盖,不累加)。"""
        if not self.cfg.use_tendency:
            return meaning
        return meaning + amount.unsqueeze(-1) * change

    def _attend(self, meaning, mask, attention, now):
        """注意力完整跑一次。返回 (特征, 改动, 权重表的键, 每个词的改动幅度, 修正后的词, 起点)。

        改动幅度由权重表定。independent_amount 打开时每个词各自定(0~1,不互相抢,需要上下文的词可以都改满);
        关掉是旧做法:权重表加起来是 1,幅度 = 权重 ÷ 最大权重,最受重视的改满,其余按比例。
        2026-11-03 第六次训练查出旧做法在读和改倾向分开后塌到 `<assistant>` 一个词上,其余几乎不改。

        打开 use_seed 时末尾多放一个起点,和词一起被修正。它没有自己的本意,不进改倾向的权重表,
        倾向也不按权重缩小,修正后整个就是注意力对全文的理解。特征、改动、键都多出起点那一列。
        """
        columns = meaning.size(1)
        if self.seed is not None:
            meaning = torch.cat([meaning, self.seed.expand(meaning.size(0), 1, -1)], 1)
            mask = torch.cat([mask, mask.new_ones(mask.size(0), 1)], 1)
        feature, change, keys = attention(meaning, mask, now)
        real = mask[:, :columns]
        query = attention.query(now, real.sum(1) - 1)
        scores = (keys[:, :columns] @ query.unsqueeze(-1)).squeeze(-1)
        if self.cfg.independent_amount:
            amount = attention.amount(scores, real)
        else:
            weights = attention.weigh(scores, real)
            amount = weights / weights.max(-1, keepdim=True).values.clamp_min(1e-9)
        vectors = self._overlay(meaning[:, :columns], change[:, :columns], amount)
        seed = None
        if self.seed is not None:
            seed = meaning[:, columns] + change[:, columns] if self.cfg.use_tendency else meaning[:, columns]
        return feature, change, keys, amount, vectors, seed

    # ---------- 输入时:注意力完整跑一次 ----------

    def prepare(self, tokens, mask):
        """输入时完整跑一次注意力,修正后的词各自过推理网络。返回说话的初始状态。

        起点过推理网络后直接成为意识流的第一列(第 0 拍的念头),不和输入词一起按比例平均。
        """
        batch, columns = tokens.shape
        now = self.current_time(torch.zeros(batch, dtype=torch.long, device=tokens.device))
        feature, _, keys, _, vectors, seed = self._attend(self.embed(tokens), mask, self.input_attention, now)
        lengths = mask.sum(1)
        S, D, P = self.cfg.flow_stream_dim, self.cfg.flow_dim, self.cfg.read_paths
        if not self.cfg.split_reading:
            return SpeakState(self.token_net(vectors), keys[:, :columns], mask, lengths,
                              vectors.new_zeros(batch, 0, S), vectors.new_zeros(batch, 0, D))
        keys = self.read_input.place(self.input_attention.read_key(feature[:, :columns]), absolute_positions(mask))
        beat_each, beat_keys = vectors.new_zeros(batch, 0, S), vectors.new_zeros(batch, P, 0, D // P)
        if seed is not None:
            # 起点站在说第 1 个字时站的位置,也就是最后一个输入词那里
            column = self.token_net(seed)
            beat_each = column.unsqueeze(1)
            beat_keys = self.read_self.place(self.stream_key(column).unsqueeze(1), (lengths - 1).unsqueeze(1))
        return SpeakState(self.token_net(vectors), keys, mask, lengths, beat_each, beat_keys, beat_each.size(1))

    # ---------- 说话:一次心跳 ----------

    def plan(self, state: SpeakState, beats: int, first: int = 0) -> BeatPlan:
        """开始说话前,把第 first ~ first + beats - 1 拍的当前时间、读的查询、“这拍不读”的分数一次算好。

        它们只由第几拍和站的位置决定(读的比例不看当前所想),和说了什么无关。以前每拍现算,
        一拍要多发几十个小任务,显卡大半时间在等 CPU 发任务(2026-11-03 实测一步 1.59 秒,显卡在算的只有 0.60 秒)。
        按拍拆成一串用 unbind:倒推时一次合回去,不会每拍各补一份全尺寸的零。
        打开 read_by_thought 时查询要看当前所想,只能每拍现算;旧做法这里只算时间。
        """
        steps = torch.arange(first, first + beats, device=state.lengths.device)
        now = self.current_time(steps)
        if not self.cfg.split_reading:
            return BeatPlan(first, now.unbind(0), None, None, None)
        times = now.expand(state.lengths.size(0), -1, -1)
        skip = self.read_self.skip_scores(times)
        skip = None if skip is None else skip.unbind(2)
        if self.cfg.read_by_thought:
            return BeatPlan(first, now.unbind(0), None, None, skip)
        here = state.lengths.unsqueeze(1) - 1 + steps
        return BeatPlan(first, now.unbind(0), self.read_input.query(times, here).unbind(2),
                        self.read_self.query(times, here).unbind(2), skip)

    def read(self, state: SpeakState, plan: Optional[BeatPlan] = None):
        """这一拍推理把各列合起来。返回 (合起来的结果 (批, 意识流宽度), 比例 (批, 路, 列))。

        列 = 输入词 + 自己这边的列(意识流起点 + 说过的列)。拆开读时,每一路在输入词上加起来是 1,
        在自己这边也是 1(可以选这拍不读),两边读到的相加;
        旧做法只有一路,输入词和说过的列挤在一张表里分,这张表就是改倾向的权重表。
        plan 是开始说话前算好的每拍时间和查询;不给就只算这一拍的,结果一样。
        """
        own_columns = state.beat_each.size(1)
        spoken = own_columns - state.start
        if plan is None:
            plan = self.plan(state, 1, first=spoken)
        beat = spoken - plan.first
        now = plan.now[beat].expand(state.lengths.size(0), -1)
        latest = state.lengths - 1 + spoken
        if not self.cfg.split_reading:
            query = self.input_attention.query(now, latest).unsqueeze(-1)
            scores = torch.cat([(state.keys @ query).squeeze(-1), (state.beat_keys @ query).squeeze(-1)], 1)
            mask = torch.cat([state.mask, state.mask.new_ones(state.mask.size(0), own_columns)], 1)
            weights = self.input_attention.weigh(scores, mask)
            columns = state.mask.size(1)
            merged = (weights[:, None, :columns] @ state.each).squeeze(1) + (weights[:, None, columns:] @ state.beat_each).squeeze(1)
            return merged, weights.unsqueeze(1)
        now, here = now.unsqueeze(1), latest.unsqueeze(1)
        if plan.input_query is not None:
            input_query, self_query = plan.input_query[beat].unsqueeze(2), plan.self_query[beat].unsqueeze(2)
        else:  # 读的比例看当前所想:最新的一列,上一拍的意识流 + 刚说出口的字
            thought = state.beat_each[:, -1:] if own_columns else None
            input_query, self_query = self.read_input.query(now, here, thought), self.read_self.query(now, here, thought)
        skip = None if plan.skip is None else plan.skip[beat].unsqueeze(2)
        merged, weights = self.read_input.read(input_query, state.keys, state.each, state.mask)
        if own_columns:
            everything = state.mask.new_ones(state.mask.size(0), own_columns)
            own, own_weights = self.read_self.read(self_query, state.beat_keys, state.beat_each, everything, skip)
            merged, weights = merged + own, torch.cat([weights, own_weights], -1)
        return merged[:, 0], weights[:, :, 0]

    def speak(self, state: SpeakState, plan: Optional[BeatPlan] = None):
        """一次心跳。返回 (字的分数, 这一拍的意识流, 这一拍读的比例 (批, 路, 列))。"""
        merged, weights = self.read(state, plan)
        stream = self.think(merged)
        return self.translate(stream), stream, weights

    def hear(self, words):
        """说出口的字:静态意思过推理网络,和输入词过的是同一个。words: (批,) 或 (批, 拍数)。"""
        return self.token_net(self.embed(words))

    def grow(self, state: SpeakState, stream, word, heard=None):
        """这一拍说完:意识流原样带上,再加上说出口的那个字,成为新的一列。

        说出口的字用静态意思,和输入词过同一个推理网络,不跑注意力。
        heard:训练时说出口的字就是答案,事先知道,可以先把整段答案一起 hear 好传进来,结果一样。
        """
        column = self.stream_gain * stream
        if self.cfg.use_spoken_word:
            column = column + (self.hear(word) if heard is None else heard)
        position = (state.lengths + state.beat_each.size(1) - state.start).unsqueeze(1)
        if self.cfg.split_reading:
            key = self.read_self.place(self.stream_key(column).unsqueeze(1), position)
            beat_keys = torch.cat([state.beat_keys, key], 2)
        else:
            key = self.input_attention.place(self.stream_key(column).unsqueeze(1), position)
            beat_keys = torch.cat([state.beat_keys, key], 1)
        return state._replace(beat_each=torch.cat([state.beat_each, column.unsqueeze(1)], 1), beat_keys=beat_keys)

    @staticmethod
    def cut_backprop(state: SpeakState):
        """误差不再沿意识流往更早的心跳传。输入部分不受影响。"""
        return state._replace(beat_each=state.beat_each.detach(), beat_keys=state.beat_keys.detach())

    def forward(self, tokens, mask):
        """从零算出第一次心跳说什么。主要给测试和简单场景用。"""
        logits, stream, _ = self.speak(self.prepare(tokens, mask))
        return logits, stream

    # ---------- 检查 ----------

    def review(self, tokens, mask, words, word_mask):
        """检查:输入和翻译出的字都用静态意思,注意力完整跑一次。

        给每个字一个置信度,并给全文覆盖一次倾向。只看字,不看意识流。
        """
        meaning = self.embed(torch.cat([tokens, words], 1))
        full_mask = torch.cat([mask, word_mask], 1)
        lengths = mask.sum(1)
        now = self.current_time(word_mask.sum(1))
        feature, _, keys, weights, vectors, seed = self._attend(meaning, full_mask, self.review_attention, now)
        columns = meaning.size(1)
        confidence = torch.sigmoid(self.confidence_head(feature[:, tokens.size(1) : columns])).squeeze(-1)
        if not self.cfg.split_reading:
            return Review(confidence, vectors, keys[:, :columns], full_mask, lengths, weights)
        keys = self.read_input.place(self.review_attention.read_key(feature[:, :columns]), absolute_positions(full_mask))
        if seed is not None:
            # 检查后的起点放在最后一列,填空时和翻译出的字一起读;位置和说话时的起点一样
            seed_key = self.read_input.place(self.review_attention.read_key(feature[:, columns:]), (lengths - 1).unsqueeze(1))
            keys = torch.cat([keys, seed_key], 2)
            vectors = torch.cat([vectors, seed.unsqueeze(1)], 1)
            full_mask = torch.cat([full_mask, full_mask.new_ones(full_mask.size(0), 1)], 1)
        return Review(confidence, vectors, keys, full_mask, lengths, weights)

    def fill(self, review: Review, blanks):
        """把空重新填。blanks: (批, 空数) 要填的是第几个字,-1 表示没有。返回 (批, 空数, 词表) 的分数。

        推理读覆盖倾向后的全文,前后的字都在;用那个字当初说出时那一拍的时间、站在说它时站的位置去读。
        和说话用的是同一个推理网络、同样的两组读法(原文一组,翻译出的字一组)和语言组件。
        """
        each = self.token_net(review.vectors)
        index = blanks.clamp_min(0)
        now = self.current_time(index)
        if not self.cfg.split_reading:
            query = self.review_attention.query(now, review.lengths.unsqueeze(1) + index)
            weights = self.review_attention.weigh(query @ review.keys.transpose(1, 2), review.mask.unsqueeze(1))
            return self.translate(self.think(weights @ each))
        # 输入一组;翻译出的字和检查后的起点(如果有)一组
        inputs = review.mask.size(1) - review.confidence.size(1) - (1 if self.seed is not None else 0)
        here = review.lengths.unsqueeze(1) - 1 + index
        thought = None
        if self.cfg.read_by_thought:  # 站在那个字上:那个字检查后的意思
            words = each[:, inputs : inputs + review.confidence.size(1)]
            thought = words.gather(1, index.unsqueeze(-1).expand(-1, -1, words.size(-1)))
        merged, _ = self.read_input.read(self.read_input.query(now, here, thought), review.keys[:, :, :inputs],
                                         each[:, :inputs], review.mask[:, :inputs])
        own, _ = self.read_self.read(self.read_self.query(now, here, thought), review.keys[:, :, inputs:],
                                     each[:, inputs:], review.mask[:, inputs:], self.read_self.skip_scores(now))
        return self.translate(self.think(merged + own))


def build_flow(cfg: ModelConfig) -> FlowModel:
    return FlowModel(cfg)
