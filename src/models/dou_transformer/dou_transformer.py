import math

import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """
    Represents a single attention head within a multi-head attention mechanism.
    """

    def __init__(self, d_model=768, n_head=8):
        super().__init__()
        self.d_head = d_model // n_head

        self.q = nn.Linear(d_model, self.d_head)
        self.k = nn.Linear(d_model, self.d_head)
        self.v = nn.Linear(d_model, self.d_head)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dim_k = torch.tensor(k.size(-1), dtype=torch.float32)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(dim_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, axis=-1)
        output = torch.bmm(attn_weights, v)
        return output

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): query embeddings.
            k (torch.Tensor): key embeddings.
            v (torch.Tensor): value embeddings.
            mask (torch.Tensor): attention mask.
        """
        output = self.scaled_dot_product_attention(
            self.q(q),
            self.k(k),
            self.v(v),
            mask=mask
        )
        return output


class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    """

    def __init__(self, d_model=768, n_head=8):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.heads = nn.ModuleList([AttentionHead(d_model=d_model, n_head=n_head) for _ in range(n_head)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        attn_outputs = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        output = self.linear(attn_outputs)
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Implements the PositionWiseFeedForward layer.
    """

    def __init__(self, d_model=768, d_ff=2048, drop=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.ff(x)


class PositionalEncoding(nn.Module):
    """
    Implements the PositionalEncoding layer.
    """

    def __init__(self, d_model=768, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, self.d_model, 2) / self.d_model)

        pe = torch.zeros(1, self.max_seq_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    """
    Implements a single Encoder layer.
    """

    def __init__(self, d_model=768, n_head=8, d_ff=2048, drop=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)

        self.norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop=drop)

        self.dropout = nn.Dropout(drop)

    def forward(self, x, mask=None):
        attn_outputs = self.masked_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_outputs)

        output = x + self.dropout(self.feed_forward(self.norm_2(x)))
        return output


class Encoder(nn.Module):
    """
    Implements the Encoder stack.
    """

    def __init__(self, vocab_size, d_model=768, n_head=8, d_ff=2048, n_layer=1, max_seq_len=512, drop=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, drop=drop) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    """
    Implements a single Decoder layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """

    def __init__(self, d_model=768, n_head=8, d_ff=2048, drop=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.masked_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)

        self.norm_2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)

        self.norm_3 = nn.LayerNorm(d_model)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop=drop)

        self.dropout = nn.Dropout(drop)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm_1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm_2(x + self.dropout(attn_output))

        output = self.norm_3(x + self.dropout(self.feed_forward(x)))
        return output


class Decoder(nn.Module):
    """
    Implements the Decoder stack.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """

    def __init__(self, vocab_size, d_model=768, n_head=8, d_ff=2048, n_layer=1, max_seq_len=512, drop=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, drop=drop) for _ in range(n_layer)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class DualTransformerWithCrossAttention(nn.Module):
    """
    Implements the Transformer architecture.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """

    def __init__(self, e_vocab_size, d_vocab_size, d_model=768, n_head=8, d_ff=2048, n_layer=1, max_seq_len=512, drop=0.1):
        super().__init__()

        self.encoder = Encoder(e_vocab_size, d_model=d_model, n_head=n_head, d_ff=d_ff, n_layer=n_layer, max_seq_len=max_seq_len, drop=drop)
        self.decoder = Decoder(d_vocab_size, d_model=d_model, n_head=n_head, d_ff=d_ff, n_layer=n_layer, max_seq_len=max_seq_len, drop=drop)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, code_input_ids, past_input_ids, code_attention_mask=None, past_attention_mask=None):
        enc_output = self.encoder(code_input_ids, mask=code_attention_mask)
        dec_output = self.decoder(past_input_ids, enc_output, src_mask=code_attention_mask, tgt_mask=past_attention_mask)
        pooled_output = dec_output[:, 0, :]
        outputs = self.regression_head(pooled_output)
        return outputs


# if __name__ == '__main__':
#     e_vocab_size, d_vocab_size = 32, 32
#     model = DualTransformerWithCrossAttention(
#         e_vocab_size=32, d_vocab_size=32, d_model=768, n_head=8, d_ff=2048, n_layer=1, max_seq_len=512, drop=0.1
#     )

#     code_input_ids = torch.randint(0, e_vocab_size, (10, 32))
#     code_attention_mask = torch.ones_like(code_input_ids)
#     past_input_ids = torch.randint(0, d_vocab_size, (10, 32))
#     past_attention_mask = torch.ones_like(past_input_ids)

#     # labels = torch.randn((code_input_ids.size(0), 1))

#     outputs = model(code_input_ids, past_input_ids, code_attention_mask=code_attention_mask, past_attention_mask=past_attention_mask)
#     print(outputs)