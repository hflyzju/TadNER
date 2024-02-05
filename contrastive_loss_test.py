import torch
import torch.nn.functional as F


# https://www.kaggle.com/code/hxshine/calculate-type-aware-contrastive-loss?scriptVersionId=160477714
def calculate_type_aware_contrastive_loss(words_emb, words_corresponding_label_emb, label_ids):
    # 输入单词的数量
    num_words = len(label_ids)

    # 创建掩码以识别匹配标签（正对）: 找到batch内label相同的数据，n*n
    pos_words_labels = torch.eq(label_ids.unsqueeze(1).repeat(1, num_words),
                                label_ids.unsqueeze(0).repeat(num_words, 1)
                                ).float()#.to(args.device)

    # 将标签嵌入与词嵌入以两种不同顺序连接
    labels_words_emb = torch.cat((words_corresponding_label_emb, words_emb), dim=-1)
    words_labels_emb = torch.cat((words_emb, words_corresponding_label_emb), dim=-1)

    # 通过矩阵乘法计算对数并进行归一化
    logits = torch.matmul(labels_words_emb, words_labels_emb.T)
    logits = F.normalize(logits, p=2, dim=0)

    # 对对数应用温度缩放
    logits = logits / torch.tensor(0.05)

    # 对数的softmax和log softmax
    softmax_logits = torch.softmax(logits, dim=-1)
    log_softmax_logits = torch.log(softmax_logits)

    # 计算对比损失
    lines_loss = -torch.mean(log_softmax_logits * pos_words_labels, dim=-1)
    loss = torch.sum(lines_loss)

    return loss


# Mock data
words_emb = torch.rand(10, 256)  # 10 words with 256-dim embeddings
words_corresponding_label_emb = torch.rand(10, 256)  # 256-dim embeddings for corresponding labels
label_ids = torch.randint(0, 5, (10,))  # Random label IDs for 10 words, assuming 5 different labels

# Function call
loss = calculate_type_aware_contrastive_loss(words_emb, words_corresponding_label_emb, label_ids)
print("Calculated Loss:", loss)
