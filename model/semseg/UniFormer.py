import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule



class Attn(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels
        self.q_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_heads)
        self.softmax = Softmax(dim=3)

        self.out = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        self.pseudo_label = None

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):

        _, N, C = emb.size()

        q_u = self.q_u(emb)
        k_u = self.k_u(emb)
        v_u = self.v_u(emb)


        # convert to multi-head representation
        mh_q_u = self.multi_head_rep(q_u).transpose(-1, -2)
        mh_k_u = self.multi_head_rep(k_u)
        mh_v_u = self.multi_head_rep(v_u).transpose(-1, -2)

        self_attn = torch.matmul(mh_q_u, mh_k_u)

        self_attn = self.attn_dropout(self.softmax(self.psi(self_attn)))
        self_attn = torch.matmul(self_attn, mh_v_u)

        self_attn = self_attn.permute(0, 3, 2, 1).contiguous()
        new_shape = self_attn.size()[:-2] + (self.KV_size,)
        self_attn = self_attn.view(*new_shape)

        out = self.out(self_attn)
        out = self.proj_dropout(out)

        return out

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_channels, channel_num, channel_num_out,
                 attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num, embedding_channels, kernel_size=1, padding=0),
                                     nn.GELU())
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = Attn(num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num)
        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                     nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en):
        if not self.training:
            en = torch.cat((en, en))

        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)

        emb = self.attn_norm(en)
        emb = self.attn(emb)
        emb = emb + en

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        if not self.training:
            out = torch.split(out, out.size(0) // 2, dim=0)[0]

        return out




class SemiDecoder(nn.Module):
    def __init__(self, num_heads, num_class, in_planes, image_size, warmup_epoch, embedding_dim):
        super(SemiDecoder, self).__init__()

        self.pseudo_label = None
        self.pseudo_prob_map = None
        self.using_SMem = False
        self.warmup_epoch = warmup_epoch

        # self.selfattention1 = SelfAttention(num_heads=num_heads,
        #                                     embedding_channels=in_planes[0],
        #                                     channel_num=in_planes[0],
        #                                     channel_num_out=in_planes[0],
        #                                     attention_dropout_rate=0.1,
        #                                     patch_num=(image_size // 4 + 1) ** 2,
        #                                     num_class=num_class)
        #
        # self.selfattention2 = SelfAttention(num_heads=num_heads,
        #                                     embedding_channels=in_planes[1],
        #                                     channel_num=in_planes[1],
        #                                     channel_num_out=in_planes[1],
        #                                     attention_dropout_rate=0.1,
        #                                     patch_num=(image_size // 8 + 1) ** 2,
        #                                     num_class=num_class)

        # self.selfattention3 = SelfAttention(num_heads=num_heads,
        #                                     embedding_channels=in_planes[2],
        #                                     channel_num=in_planes[2],
        #                                     channel_num_out=in_planes[2],
        #                                     attention_dropout_rate=0.1,
        #                                     patch_num=(image_size // 16 + 1) ** 2,
        #                                     num_class=num_class)

        self.selfattention4 = SelfAttention(num_heads=num_heads,
                                            embedding_channels=in_planes[3],
                                            channel_num=in_planes[3],
                                            channel_num_out=in_planes[3],
                                            attention_dropout_rate=0.1,
                                            patch_num=(image_size // 32 + 1) ** 2,
                                            num_class=num_class)

        self.decoder = SegFormerHead(embedding_dim, in_planes, num_class)


    def set_pseudo_label(self, pseudo_label):
        self.pseudo_label = pseudo_label

    def set_pseudo_prob_map(self, pseudo_prob_map):
        self.pseudo_prob_map = pseudo_prob_map

    def forward(self, feats, h, w):
        e1, e2, e3, e4 = feats

        e4_selfattention = self.selfattention4(e4)

        out = self.decoder(e1, e2, e3, e4_selfattention)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out



class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, embedding_dim, in_channels, num_class):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels


        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(0.1)

        self.linear_pred = nn.Conv2d(embedding_dim, num_class, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
