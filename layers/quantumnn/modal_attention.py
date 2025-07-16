class ModalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, modal_reps):
        """
        输入:
        modal_reps: List[Tensor], 每个tensor shape为[batch_size, seq_len, embed_dim]
        
        输出:
        weights: [batch_size, seq_len, num_modalities]
        """
        batch_size, seq_len, _ = modal_reps[0].shape
        num_modalities = len(modal_reps)
        
        # 堆叠所有模态
        # [batch_size, seq_len, num_modalities, embed_dim]
        stacked_reps = torch.stack(modal_reps, dim=2)
        
        # 投影并分头
        # [batch_size, seq_len, num_modalities, num_heads, head_dim]
        Q = self.q_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        K = self.k_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        V = self.v_proj(stacked_reps).view(batch_size, seq_len, num_modalities, self.num_heads, self.head_dim)
        
        # 转置以便进行注意力计算
        Q = Q.transpose(2, 3)  # [batch_size, seq_len, num_heads, num_modalities, head_dim]
        K = K.transpose(2, 3)
        V = V.transpose(2, 3)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头的结果并计算最终权重
        # [batch_size, seq_len, num_modalities]
        modal_weights = attn_weights.mean(dim=2)  # 平均所有头的权重
        
        return modal_weights 