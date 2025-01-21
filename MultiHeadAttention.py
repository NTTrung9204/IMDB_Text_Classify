import torch
import torch.nn as nn
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_embedding, s_length, num_heads, vocab_size):
        super(MultiHeadAttention, self).__init__()
        self.d_embedding = d_embedding
        self.s_length = s_length
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        assert d_embedding % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.head_dim = d_embedding // num_heads
        self.query = nn.Linear(d_embedding, d_embedding)
        self.key = nn.Linear(d_embedding, d_embedding)
        self.value = nn.Linear(d_embedding, d_embedding)
        self.W_0 = nn.Linear(d_embedding, d_embedding)

        self.ff_layer_1 = nn.Linear(d_embedding, 4 * d_embedding)
        self.ff_layer_2 = nn.Linear(4 * d_embedding, d_embedding)

        self.relu = nn.ReLU(True)
        self.drop = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(d_embedding)

        self.fc_1 = nn.Linear(d_embedding, 32)
        self.fc_2 = nn.Linear(32, 2)

        self.embedding_layer = nn.Embedding(vocab_size, d_embedding)
        self.embedding_pos = nn.Embedding(s_length, d_embedding)

    def _attention_block(self, x):
        batch_size = x.size(0)

        Q = self.query(x)  # (batch_size, seq_length, d_embedding)
        K = self.key(x)    # (batch_size, seq_length, d_embedding)
        V = self.value(x)  # (batch_size, seq_length, d_embedding)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_length, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_length, seq_length)
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_length, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, -1, self.d_embedding)  # (batch_size, seq_length, d_embedding)

        output = self.W_0(attn_output)  # (batch_size, seq_length, d_embedding)

        output = self.layer_norm(output + x)  # (batch_size, seq_length, d_embedding)

        return output

    def _feed_forward_block(self, x):
        x = self.ff_layer_1(x)
        x = self.drop(x)
        x = self.relu(x)

        x = self.ff_layer_2(x)
        x = self.drop(x)

        return x

    def forward(self, x):
        x = x.to(self.embedding_layer.weight.device).long()
        emb_layer = self.embedding_layer(x)  # (batch_size, seq_length, d_embedding)

        seq_length = x.size(1)
        position_indices = torch.arange(0, seq_length, device=x.device).unsqueeze(0)  # (1, seq_length)
        emb_pos = self.embedding_pos(position_indices)  # (1, seq_length, d_embedding)

        x = emb_layer + emb_pos  # (batch_size, seq_length, d_embedding)

        # Multi-head attention block
        x = self._attention_block(x)

        # Mean pooling và fully connected
        x = x.mean(dim=1)  # (batch_size, d_embedding)

        x = self.fc_1(x)
        x = self.drop(x)

        x = self.fc_2(x)
        x = self.drop(x)

        return x


# Transformer = MultiHeadAttention(d_embedding=128, s_length=200, num_heads=8, vocab_size=20000)
# Transformer.eval()
# inputs = torch.randint(0, 5000, (1, 200))

# print(Transformer(inputs))

# summary(Transformer, input_size=(200,), batch_size=256)

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#          Embedding-1            [256, 200, 128]       2,560,000
#          Embedding-2            [256, 200, 128]          25,600
#             Linear-3            [256, 200, 128]          16,512
#             Linear-4            [256, 200, 128]          16,512
#             Linear-5            [256, 200, 128]          16,512
#             Linear-6            [256, 200, 128]          16,512
#          LayerNorm-7            [256, 200, 128]             256
#             Linear-8            [256, 200, 128]          16,512
#             Linear-9            [256, 200, 128]          16,512
#            Linear-10            [256, 200, 128]          16,512
#            Linear-11            [256, 200, 128]          16,512
#         LayerNorm-12            [256, 200, 128]             256
#            Linear-13            [256, 200, 128]          16,512
#            Linear-14            [256, 200, 128]          16,512
#            Linear-15            [256, 200, 128]          16,512
#            Linear-16            [256, 200, 128]          16,512
#         LayerNorm-17            [256, 200, 128]             256
#            Linear-18            [256, 200, 128]          16,512
#            Linear-19            [256, 200, 128]          16,512
#            Linear-20            [256, 200, 128]          16,512
#            Linear-21            [256, 200, 128]          16,512
#         LayerNorm-22            [256, 200, 128]             256
#            Linear-23            [256, 200, 128]          16,512
#            Linear-24            [256, 200, 128]          16,512
#            Linear-25            [256, 200, 128]          16,512
#            Linear-26            [256, 200, 128]          16,512
#         LayerNorm-27            [256, 200, 128]             256
#            Linear-28            [256, 200, 128]          16,512
#            Linear-29            [256, 200, 128]          16,512
#            Linear-30            [256, 200, 128]          16,512
#            Linear-31            [256, 200, 128]          16,512
#         LayerNorm-32            [256, 200, 128]             256
#            Linear-33                  [256, 32]           4,128
#           Dropout-34                  [256, 32]               0
#            Linear-35                   [256, 2]              66
#           Dropout-36                   [256, 2]               0
# ================================================================
# Total params: 2,987,618
# Trainable params: 2,987,618
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.20
# Forward/backward pass size (MB): 1600.13
# Params size (MB): 11.40
# Estimated Total Size (MB): 1611.72
# ----------------------------------------------------------------