"""v5 的所有参数。想改设计先看这个文件。

命令行可以临时覆盖任何一项,比如:
    python train.py --model flow --attn_rounds 6
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    # flow = 意识流模型(第五版);gru、transformer = 对照模型
    model_type: str = "flow"
    vocab_size: int = 8000

    # ===== 意识流模型 =====
    flow_dim: int = 128          # 一个词的静态意思和倾向的长度,也是注意力内部的宽度
    flow_time_dim: int = 128     # 当前时间向量的长度
    flow_dials: int = 32         # 时钟表针数
    flow_heads: int = 4          # 注意力每轮分几路
    flow_stream_dim: int = 512   # 意识流的宽度,也是推理里每个词各自过的网络宽度。越大保留的信息越多
    flow_context: int = 512      # 输入最多保留多少个词,更早的看不见

    # 注意力在“输入时”跑几轮。用户定的第一档是 4 轮;2026-11-03 排查发现注意力把语序编得很弱,
    # 用户要求加层数,改成 8 轮
    attn_rounds: int = 8
    # 检查时注意力跑几轮
    review_rounds: int = 8

    # ===== 说话和填空时怎么把各列合起来(2026-11-03 改) =====
    # 打开:读原文(输入词)和读自己说过的列分开,各自的比例加起来是 1,读到的再相加;比例和“改倾向的权重表”分开。
    # 关掉:旧做法,输入词和说过的列挤在一张表里分,这张表同时决定倾向改多少。旧存档读出来按关掉算
    split_reading: bool = True
    # 每拍分几路读,每路一张比例表,读每列的一段,几路各看各的。只在 split_reading 打开时有用
    read_paths: int = 4
    # 读自己说过的那一份,每一路多一个“这拍不读”的选项。原文那一份没有这个选项,每拍一定读。
    # 小任务上不加这个时,复述明显学得慢:用不上自己说过的字时也被迫读一半
    self_can_skip: bool = True
    # 意识流起点:输入末尾放一个空位,和词一起被注意力修正,修正后直接成为意识流的第一列(第 0 拍的念头),
    # 不和输入词一起按比例平均。检查时也有一个,填空时读。2026-11-03 排查发现语序在第 1 拍按比例平均时
    # 从 0.40 掉到 0.15,用户同意加。只在 split_reading 打开时有用,旧存档读出来按关掉算。
    # 第六次训练 1 小时没起作用(模型已经把 <assistant> 当成装整段理解的词),用户说关掉,开关留着
    use_seed: bool = False
    # 每个词的改动幅度各自定:权重表给每个词一个 0~1 的幅度,词和词不互相抢,需要上下文的词可以都改满。
    # 关掉是旧做法:表加起来是 1,幅度 = 权重 ÷ 最大权重。第六次训练查出旧做法塌到 <assistant> 一个词上,
    # 其余词幅度不到 0.01,等于没被修正。旧存档读出来按关掉算
    independent_amount: bool = True
    # 读的比例也看“当前在想什么”(最新一列:上一拍的意识流 + 刚说出口的字)。默认关:用户认为推理不需要
    # 内置注意力。只用来在小任务上对比,打开前要和用户商量
    read_by_thought: bool = False

    # ===== 生成时的检查和填空 =====
    max_outputs: int = 256       # 一段话最多说多少个词
    max_rework_rounds: int = 3   # 最多检查、填空几次
    rework_threshold: float = 0.5  # 置信度低于这个数的字要挖掉重填,高于的保留
    rework_max_bad: int = 16     # 一次最多填多少个空

    # 功能开关:关掉一个再从头训练,对比损失,就知道它有没有用
    use_clock: bool = True       # 时钟。关掉后当前时间一直等于初始时间
    use_order: bool = True       # 注意力看到的表征随位置变化
    use_tendency: bool = True    # 倾向。关掉后每个词就是静态意思,权重表照常
    use_review: bool = True      # 说完之后的检查和填空
    # 每拍加进上下文的新一列,在意识流之外再带上说出口的那个字(训练时用答案的字,生成时用自己说的字)。
    # 关掉后只接意识流:2026-11-03 第一版就是这样,真实文本上学不会字和字怎么接
    use_spoken_word: bool = True

    # ===== 对照模型 =====
    embed_dim: int = 256
    gru_hidden: int = 1000
    tf_layers: int = 5
    tf_heads: int = 8
    tf_dim: int = 256
    tf_context: int = 384


@dataclass
class TrainConfig:
    # 每步同时走多少段话。说话一拍一拍按顺序走,显卡吃不饱,16 段和 32 段每步耗时几乎一样,
    # 32 段显存峰值约 5 GB(答案 256、输入 512),给其他程序留出余量
    beat_batch: int = 32
    batch_size: int = 8
    window: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay_tokens: int = 0
    max_tokens: int = 0          # 算够多少个预测就停(一次心跳说一个词算一个预测)
    max_minutes: float = 0.0
    log_every: int = 20
    eval_every_tokens: int = 1_000_000
    eval_beats: int = 20_000     # 每次验证算多少个预测
    seed: int = 1234
    data_dir: str = "data"
    run_dir: str = "runs"
    run_name: str = ""
    resume: bool = False

    # ===== 第五版训练办法 =====
    # 训练时一段答案最多多长。更长的截掉,截掉的不补停止符
    max_answer: int = 256
    # 误差沿着意识流最多往回传多少次心跳。0 = 一直传到这段话开头
    backprop_beats: int = 0
    # 训练时模型自己选字的方式,和生成一样。选出的字交给检查,和答案比对错
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    # 教检查时,说对的字和说错的字各占一半分量。不这样的话,大部分字都对(或都错)时,
    # 检查只学“全说可信”(或“全说不可信”)最划算,不去一个个分辨
    balance_check: bool = True
    # 检查和填空的损失占多大比重
    confidence_loss_weight: float = 0.5
    fill_loss_weight: float = 0.5
