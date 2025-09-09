## Reference
Code taken directly from https://github.com/juho-lee/set_transformer (Code accessed 14.03.2025)
Based on the SetTransformer Architecture of https://arxiv.org/abs/1810.00825


@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}


Notes: 
- Look into DeepSets
- Can the set transformer handle training with x number of agents and then y number of agents?
  - https://github.com/juho-lee/set_transformer/issues/12#issue-942935255
  - see implementation of MAB with multi-head attention
  - "It is worth noticing that masked self-attention is not permutation equivariant"
  - from the issue:
```python3
class MAB(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ln: bool = False):
      super(MAB, self).__init__()
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
      if ln:
          self.ln0 = nn.LayerNorm(embed_dim)
          self.ln1 = nn.LayerNorm(embed_dim)
    
      self.fc_o = nn.Linear(embed_dim, embed_dim)
    
    def forward(
      self,
      X,
      Y,
      key_padding_mask: Optional[torch.Tensor] = None,
      attn_mask: Optional[torch.Tensor] = None,
    ):
      H = X + self.multihead_attn(
          X, Y, Y, key_padding_mask=key_padding_mask, attn_mask=attn_mask
      )
      H = H if getattr(self, 'ln0', None) is None else self.ln0(H)
      O = H + F.relu(self.fc_o(H))
      O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
      return O
```
