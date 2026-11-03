"""分词器:把文字切成 token。

用 BPE 算法:先把常用字收进词表,再把最常连在一起的字合并成词,直到词表满。
"""
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

# 特殊 token 的编号固定,模型代码直接用
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2  # 一段对话或文本的开头,读到它时模型忘掉之前的对话
EOS_ID = 3  # 一段对话或文本的结尾
USER_ID = 4  # 接下来是用户说的话
ASSISTANT_ID = 5  # 接下来是模型的回答
SYSTEM_ID = 6  # 接下来是系统提示
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<system>"]


def train_tokenizer(texts, path, vocab_size=8000, limit_alphabet=6000):
    """texts:可迭代的字符串。limit_alphabet:最多收多少个单字,剩下的名额留给合并出来的词。"""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    # 按连续的字、连续的空白、单个符号切开,保留所有字符,解码时才能原样拼回去
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s+|[^\w\s]"), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        limit_alphabet=limit_alphabet,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def decode_token(tokenizer: Tokenizer, token_id: int) -> str:
    """单个 token 的文字,换行等不可见字符转成看得见的形式,用于观察工具。"""
    text = tokenizer.id_to_token(token_id)
    return text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "␣")
