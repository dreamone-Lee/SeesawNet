import torch
import torch.nn as nn
from layers.Embed import newPatchEmbed, positional_encoding, newPatchEmbed1
from layers.SelfAttention_Family import TSMixer, ResAttention
from layers.SeesawNet_backbone import TSEncoder, PD_Attention_layer, CR_Attention_layer, aggregation_layer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.patch_num = int((configs.seq_len - configs.patch_len) / configs.stride) + 2
        self.aggr_num = 2 if configs.down_sample_rate == 0 else int(self.patch_num * configs.down_sample_rate)

        # Embedding
        self.pe = positional_encoding(configs.pe, configs.learn_pe, self.patch_num, configs.d_model)
        self.embedding_x = newPatchEmbed(configs, d_model=configs.d_model)
        self.embedding_xx = newPatchEmbed(configs, d_model=configs.d_model)
        
        # Encoder
        layers = self.layers_init(configs)
        self.encoder = TSEncoder(layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.aggr_num * configs.d_model, configs.pred_len, bias=True)
        )

    def layers_init(self, configs):
        PD_Attention = [PD_Attention_layer(
            self.aggr_num, configs.d_model, configs.d_ff, configs.n_heads, dropout=configs.dropout, 
            activation=configs.activation, norm=configs.norm, group=configs.group
        ) for i in range(configs.pd_layers)]
        Aggregation = aggregation_layer(
            self.patch_num, self.aggr_num, configs.d_model, configs.n_heads, dropout=configs.dropout, 
            activation=configs.activation, norm=configs.norm)
        CR_Attention = [CR_Attention_layer(
            self.c_in, configs.d_model, configs.d_ff, configs.n_heads, dropout=configs.dropout, 
            activation=configs.activation, norm=configs.norm, group=configs.group
        ) for i in range(configs.cr_layers)]
        return [Aggregation, *CR_Attention, *PD_Attention]


    def forward(self, x, x_mark=None):
        """
        input: 
            x: [B, L, N], x_mark: [B, L, 4] (optional)
            where L = seq_len, N = c_in
        
        z: [B, N, P, D]
        zz: [B, N, P, D]
        enc_out: [B, N, P, D]

        output: 
            dec_out: [B, L1, N], where L1 = pred_len
        """ 
        if x_mark is None:
            x_mark = torch.zeros((*x.shape[:-1], 4), device=x.device)
        xx = x.clone() # nonstationary input

        mean = x.mean(1, keepdim=True).detach()
        x = x - mean
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / std # stationary input

        z = self.embedding_x(x, x_mark, self.pe)
        zz = self.embedding_xx(xx, x_mark, self.pe)
        
        enc_out = self.encoder(z, zz)[0][:, :self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)
        dec_out = dec_out * (std + 1e-5) + mean

        return dec_out[:, -self.pred_len:, :], std, mean  # [B, L_pred, N]
