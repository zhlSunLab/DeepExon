import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout_rate):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        with open('/home/yzq/DeepCellEss-main_shaungjiehe/DeepCellEss-main/data/Amino_acid_position_probability.pkl', 'rb') as file:
            amino_pos_probs = pickle.load(file)
        if not isinstance(amino_pos_probs, torch.Tensor):
            amino_pos_probs = torch.tensor(amino_pos_probs, dtype=torch.float)
        self.register_buffer('amino_pos_probs', amino_pos_probs)
        with open('/home/yzq/DeepCellEss-main_shaungjiehe/DeepCellEss-main/data/Conditional_dipeptide_probability.pkl', 'rb') as file:
            self.dipeptide_probabilities = pickle.load(file)
        dipeptide_lookup_table = torch.zeros(22, 22)
        for first_aa_index, inner_dict in self.dipeptide_probabilities.items():
            for (idx1, idx2), probability in inner_dict.items():
                dipeptide_lookup_table[idx1, idx2] = probability
        self.register_buffer('dipeptide_lookup_table', dipeptide_lookup_table)

    def forward(self, x, x_1, EXON_POS):
        # x_1
        batch_size, seq_len = x_1.size()
        output_tensor = ((x_1 >= 1) & (x_1 <= 21)).unsqueeze(-1).float()
        # x
        tok_embedding = self.tok_embed(x)
        batch_size, seq_len = x.size()
        probs = torch.zeros(batch_size, seq_len, device=x.device)
        mask = (x != 0)
        mask_1 = mask.unsqueeze(-1).expand(-1, -1, self.d_model)
        pe = self.pe[:seq_len]
        pe = pe.masked_fill(mask_1 == 0, 0)
        for i in range(seq_len):
            probs[:, i] = self.amino_pos_probs[x[:, i], i]
        idx1s = x[:, :-1]
        idx2s = x[:, 1:]
        dipeptide_probs = self.dipeptide_lookup_table[idx1s, idx2s].unsqueeze(-1)
        zero_padding = torch.zeros(batch_size, 1, 1, device=x.device)
        dipeptide_probs_padded = torch.cat([dipeptide_probs, zero_padding], dim=1)
        probs_expanded = probs.unsqueeze(-1)
        pe_adjusted = pe * (probs_expanded + dipeptide_probs_padded)
        embedding = tok_embedding + pe_adjusted
        embedding = self.dropout(embedding)
        embedding_EXON = torch.cat([output_tensor, EXON_POS], dim=-1)

        return embedding, embedding_EXON


class MSCNN(nn.Module):
    def __init__(self, fea_size):
        super().__init__()
        self.layerNorm = nn.LayerNorm(fea_size)
        self.conv_2 = nn.Sequential(nn.Conv1d(in_channels=fea_size, out_channels=fea_size, kernel_size=2, padding='same'),
                               nn.GELU())
        self.conv_3 = nn.Conv1d(in_channels=fea_size, out_channels=fea_size, kernel_size=3, padding='same')

    def forward(self, x):
        x_1 = self.layerNorm(x.permute(0, 2, 1))
        x = x_1.permute(0, 2, 1)
        conv_2 = self.conv_2(x)
        x = conv_2 * self.conv_3(x)

        return x


class DeepEXON(nn.Module):
    def __init__(self, args, max_len, fea_size, kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop,
                 linear_drop,
                 structure='TextCNN+MultiheadAttn+BiLSTM+Maxpool+MLP', name='DeepEXON'):
        super(DeepEXON, self).__init__()
        self.structure = structure
        self.name = name
        fea_size = 120
        self.embedding = TransformerEmbedding(22, 30, max_len, dropout_rate=0.2)
        self.mscnn = MSCNN(30)
        self.multiAttn = nn.MultiheadAttention(embed_dim=fea_size,
                                               num_heads=num_head,
                                               dropout=attn_drop,
                                               batch_first=True)

        self.textCNN = nn.Conv1d(in_channels=22,
                                 out_channels=22,
                                 kernel_size=kernel_size,
                                 padding='same')
        self.textCNN_EXON = nn.Sequential(nn.Conv1d(in_channels=22, out_channels=22, kernel_size=1, padding='same'),
                                             nn.ReLU(),
                                             nn.Conv1d(in_channels=22, out_channels=90, kernel_size=3, padding='same'),
                                             nn.ReLU())
        self.layerNorm = nn.LayerNorm(fea_size)
        self.biLSTM = nn.LSTM(fea_size,
                              42,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=num_layers,
                              dropout=lstm_drop)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.generator = nn.Sequential(nn.Linear(42 * 2, 1),
                                       nn.Dropout(linear_drop),
                                       nn.Sigmoid())

    def forward(self, x, x_EXON, X_EXON_POS, get_attn=False):
        x_c, x_w = self.embedding(x, x_EXON, X_EXON_POS)
        # x_c
        residual = x_c
        x_c = x_c.permute(0, 2, 1)
        # => batch_size × fea_size × seq_len
        x_c = F.relu(self.cnnatt(x_c))
        x_c = residual + x_c.permute(0, 2, 1)
        # x_w
        x_w = x_w.permute(0, 2, 1)
        # => batch_size × fea_size × seq_len
        x_w = self.textCNN_EXON(x_w)
        x_w = x_w.permute(0, 2, 1)

        x_c = torch.cat([x_c, x_w], dim=-1)

        attn_output, seq_attn = self.multiAttn(x_c, x_c, x_c)
        x_c = x_c + self.layerNorm(attn_output)
        x_c, _ = self.biLSTM(x_c)
        x_c = x_c.permute(0, 2, 1)
        x_c = self.generator(self.pool(x_c).squeeze(-1))
        if get_attn == True:
            return x_c
        else:
            return x_c
