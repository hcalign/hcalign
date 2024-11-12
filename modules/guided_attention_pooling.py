import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedAttentionPool(nn.Module):
    def __init__(self, embed_dim=512, dropout=0.3):
        super(GuidedAttentionPool, self).__init__()
        self.embed_dim = embed_dim
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_embeds, ref_embeds):
        """
        Args:
            query_embeds: (N, D)
            ref_embeds: (M, T, D)
        Returns:
            attn_out: (M, N, M)
        """
        query_embeds = self.layer_norm1(query_embeds)
        ref_embeds = self.layer_norm1(ref_embeds)
        attention_logits = ref_embeds @ query_embeds.permute(1, 0) # (M, T, N)
        attention_logits = attention_logits / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_logits, dim=1) # (M, T, N)
        attention = ref_embeds.permute(0, 2, 1) @ attention_weights # (M, D, N)
        attention = attention.permute(0, 2, 1)
        attn_out = self.layer_norm2(attention)
        attention = self.dropout(attn_out)
        return attn_out


