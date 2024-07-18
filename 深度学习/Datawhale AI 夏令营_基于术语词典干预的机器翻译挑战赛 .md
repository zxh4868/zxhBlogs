## 基于术语词典干预的机器翻译

在baseline的基础上添加了soft attention，当N=2000时，没有问题，但是一旦增加数据量就会爆显存，还需要找一下问题

完整代码如下

```python
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from keras import device
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
import random
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
import time
from sacrebleu.metrics import BLEU
from tqdm import tqdm


# 数据准备
# 修改TranslationDataset类以处理数据
class TranslationDataset(Dataset):
    def __init__(self, file_name, terminology):

        self.data = []
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                en, zh = line.strip().split('\t') # 训练数据en和zh使用制表符分隔\t
                self.data.append((en, zh))

        self.terminology = terminology

        # 创建词汇表，这里一定要将术语词典中的词也包含在词汇表中
        self.en_tokenizer = get_tokenizer('basic_english')
        self.zh_tokenizer = list # 使用用字符级分词

        en_vocab = Counter(self.terminology.keys()) # 确保术语在词汇表中
        zh_vocab = Counter()

        for en, zh in self.data:
            en_vocab.update(self.en_tokenizer(en))
            zh_vocab.update(self.zh_tokenizer(zh))

        # 添加术语到词汇表
        self.en_vocab = ['<pad>', '<sos>', '<eos>' ] + list(self.terminology.keys()) + [word for word, _ in en_vocab.most_common(10000)]
        self.zh_vocab = ['<pad>', '<sos>', '<eos>' ] + [word for word, _ in zh_vocab.most_common(10000)]

        # 完成单词与索引的对应
        self.en_word2idx = {word: idx for idx, word in enumerate(self.en_vocab)}
        self.zh_word2idx = {word: idx for idx, word in enumerate(self.zh_vocab)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将一条训练数据(en,zh)转化为tensor
        en, zh = self.data[idx]
        en_tensor = torch.Tensor([self.en_word2idx.get(word, self.en_word2idx['<sos>']) for word in self.en_tokenizer(en)] + [self.en_word2idx['<eos>']]).long()
        zh_tensor = torch.Tensor([self.zh_word2idx.get(word, self.zh_word2idx['<sos>']) for word in self.zh_tokenizer(zh)] + [self.zh_word2idx['<eos>']]).long()
        return en_tensor, zh_tensor

# 术语词典加载
def load_terminology_dictionary(dict_file):
    terminology = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            en_term, ch_term = line.strip().split('\t')
            terminology[en_term] = ch_term
    return terminology

def collate_fn(batch):
    """
    collate_fn 函数通常用于数据加载器（DataLoader）, 对批次数据进行处理和填充
    :param batch:
    :return:
    """
    en_batch, zh_batch = [], []
    # 遍历批次中的每个样本
    for en_item, zh_item in batch:
        en_batch.append(en_item)
        zh_batch.append(zh_item)

    # 对英文的中文序列分别进行填充, 0 对应的是 <pad>
    en_batch = nn.utils.rnn.pad_sequence(en_batch, padding_value=0, batch_first=True)
    zh_batch = nn.utils.rnn.pad_sequence(zh_batch, padding_value=0, batch_first=True)
    return en_batch, zh_batch


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : [batch_size, seq_len] => [batch_size, seq_len, emb_dim]
        embedded = self.dropout(self.embedding(x))
        # outputs : [batch_size, seq_len, hidden_dim]
        # hidden : [num_layers, batch_size, seq_len]
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(self.output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, context):
        # x : [batch_size, seq_len] => [batch_size, seq_len, emb_dim]
        embedded = self.dropout(self.embedding(x))
        # outputs : [batch_size, seq_len, hidden_dim]
        _, hidden = self.rnn(embedded, hidden)
        # print("==============", hidden.shape, context.shape)
        if hidden.shape[1] == 1:
            hc = torch.cat((context.squeeze(2), hidden[1]), dim=1)
        else:
            hc = torch.cat((context.squeeze(), hidden[1].squeeze()), dim=1)
        # 由seq_len = 1的 [batch_size, output_dim]
        pred = self.output(hc)
        return pred, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attention = nn.Linear(encoder.hidden_dim, self.decoder.hidden_dim)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # 获取上下文向量
        o, hidden = self.encoder(src)

        input = trg[:, 0].unsqueeze(1) # start token
        for t in range(1, trg_len):
            # print(hidden.shape)
            # print(o.shape)
            # [batch_size, seq, 1] = [batch_size, seq_len, hidden_dim] @ [batch_size, hidden_dim, 1]
            attn_prob = torch.matmul(self.attention(o), hidden[1].squeeze().unsqueeze(2))
            # 计算注意力权重 [batch_size, seq_len, 1]
            attn_weight = F.softmax(attn_prob, dim=1)
            # 计算上下文向量 [batch_size,hidden_dim, seq_len ] matmul [batch_size, seq_len, 1]
            context = torch.matmul(o.permute(0, 2, 1), attn_weight)
            output, hidden = self.decoder(input, hidden, context)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            result = output.argmax(dim=1) # [batch_size, 1]
            input = trg[:, t].unsqueeze(1) if teacher_force else result.detach().unsqueeze(1)
        return outputs


def train(device, model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in tqdm(enumerate(iterator)):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # 清理显存
        # del src, trg, output, loss
        # torch.cuda.empty_cache()

    return epoch_loss / len(iterator)


def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

# 更新translate_sentences函数以考虑术语词典
def translate_sentence(sentence, model, dataset : TranslationDataset, terminology, device: torch.device, max_len: int = 50 ):
    model.eval()
    tokens = dataset.en_tokenizer(sentence)
    tensor = torch.LongTensor([dataset.en_word2idx.get(token, dataset.en_word2idx['<sos>']) for token in tokens]).unsqueeze(0).to(device) # [1, seq_len]

    with torch.no_grad():
        o, hidden = model.encoder(tensor)

    translated_tokens = []
    input_token = torch.LongTensor([[dataset.zh_word2idx['<sos>']]]).to(device) # [1,1]

    for i in range(max_len):
        # print(hidden.shape)
        # print("model.attention(o) shape:", model.attention(o).shape)
        # print("hidden[1] shape before squeeze and unsqueeze:", hidden.shape)
        # print("hidden[1] shape after squeeze and unsqueeze:", hidden[1].squeeze(0).unsqueeze(2).shape)
        attn_prob = torch.matmul(model.attention(o), hidden[1].unsqueeze(2))
        # 计算注意力权重 [batch_size, seq_len, 1]
        attn_weight = F.softmax(attn_prob, dim=1)
        # 计算上下文向量 [batch_size,hidden_dim, seq_len ] matmul [batch_size, seq_len, 1]
        context = torch.matmul(o.permute(0, 2, 1), attn_weight)
        output, hidden = model.decoder(input_token, hidden, context)
        result = output.argmax(dim=1)
        translated_token = dataset.zh_vocab[result.item()]

        if translated_token == '<eos>':
            break

        # 如果翻译的词在术语词典中，则使用术语词典中的词
        if translated_token in terminology.values():
            for en_term, ch_term in terminology.items():
                if translated_token == ch_term:
                    translated_token = en_term

        translated_tokens.append(translated_token)
        input_token = result.unsqueeze(1)
    return ''.join(translated_tokens)


def evaluate_bleu(model: Seq2Seq, dataset: TranslationDataset,src_file,ref_file,terminology ,device: torch.device):
    model.eval()

    src_sentences = load_sentences(src_file)
    ref_sentences = load_sentences(ref_file)

    translated_sentences = []

    for src in src_sentences:
        translated = translate_sentence(src, model, dataset, terminology, device)
        translated_sentences.append(translated)

    bleu = BLEU()
    score = bleu.corpus_score(translated_sentences, [ref_sentences])

    return score

def inference(model: Seq2Seq, dataset: TranslationDataset, src_file: str, save_dir: str, terminology,
              device: torch.device):
    model.eval()
    src_sentences = load_sentences(src_file)

    translated_sentences = []
    for src in src_sentences:
        translated = translate_sentence(src, model, dataset, terminology, device)
        # print(translated)
        translated_sentences.append(translated)
        # print(translated_sentences)

    # 将列表元素连接成一个字符串，每个元素后换行
    text = '\n'.join(translated_sentences)

    # 打开一个文件，如果不存在则创建，'w'表示写模式
    with open(save_dir, 'w', encoding='utf-8') as f:
        # 将字符串写入文件
        f.write(text)

    # return translated_sentences

if __name__ == '__main__':
    start_time = time.time()  # 开始计时

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # terminology = load_terminology_dictionary('../dataset/en-zh.dic')
    terminology = load_terminology_dictionary('nlp/datasets/fanyi/en-zh.dic')

    # 加载数据 nlp/datasets/fanyi/en-zh.dic
    dataset = TranslationDataset('nlp/datasets/fanyi/train.txt', terminology=terminology)
    # 选择数据集的前N个样本进行训练
    N = 2000 # 或者你可以设置为数据集大小的一定比例，如 int(len(dataset) * 0.1)
    subset_indices = list(range(N))
    subset_dataset = Subset(dataset, subset_indices)
    train_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    print(len(dataset))
    # train_loader = DataLoader(Subset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 定义模型参数
    INPUT_DIM = len(dataset.en_vocab)
    OUTPUT_DIM = len(dataset.zh_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    # 初始化模型
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    # 打印模型中的所有参数
    for name, param in model.named_parameters():
        print(f'Parameter name: {name}')
        print(f'Parameter size: {param.size()}')
        print('-------------------------')

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.zh_word2idx['<pad>'])

    # 训练模型
    N_EPOCHS = 50
    CLIP = 1

    # for epoch in range(N_EPOCHS):
    #     train_loss = train(device, model, train_loader, optimizer, criterion, CLIP)
    #     print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f}')

    # # 在训练循环结束后保存模型
    # torch.save(model.state_dict(), 'nlp/datasets/fanyi/translation_model_GRU.pth')

    end_time = time.time()  # 结束计时

    # 计算并打印运行时间
    elapsed_time_minute = (end_time - start_time) / 60
    print(f"Total running time: {elapsed_time_minute:.2f} minutes")

    # 加载训练好的模型
    model.load_state_dict(torch.load('nlp/datasets/fanyi/translation_model_GRU.pth'))

    # 评估BLEU分数
    bleu_score = evaluate_bleu(model, dataset, 'nlp/datasets/fanyi/dev_en.txt', 'nlp/datasets/fanyi/dev_zh.txt',
                               terminology=terminology, device=device)
    print(f'BLEU-4 score: {bleu_score.score:.2f}')

    # 加载训练好的模型
    model.load_state_dict(torch.load('nlp/datasets/fanyi/translation_model_GRU.pth'))

    save_dir = 'nlp/datasets/fanyi/submit.txt'
    inference(model, dataset, src_file="nlp/datasets/fanyi/test_en.txt", save_dir=save_dir, terminology=terminology,
              device=device)
    print(f"翻译完成！文件已保存到{save_dir}")

```

