
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from RDCNN import *
from MWSA import *
from EMVM import *

class Encoder(nn.Module):
    def __init__(self, base_channel, num_blocks=[4,4,4], num_heads=[4,8], mlp_ratio=2.66, window_size = 8):
        super(Encoder, self).__init__()
        self.window_size = window_size
        self.num_blocks_stage2 = num_blocks[1]
        self.num_blocks_stage3 = num_blocks[2]
        self.encoder_sampling = nn.ModuleList([
            BasicConv(base_channel, base_channel*2, kernel_size=3, act=True, stride=2, bias=False, norm=True),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, act=True, stride=2, bias=False, norm=True)
            ])
        self.cnn_stem = nn.Sequential(
            BasicConv(3, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True),
            BasicConv(base_channel, base_channel, kernel_size=3, act=True, stride=1, bias=False, norm=True),
        )
        self.ecnn_stage = nn.Sequential(
            *[
                ResidualDepthBlock(base_channel, base_channel) for i in range(num_blocks[0])
            ]
        )

        self.emix_stage1 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel*2),
                SimpleSwinBlock(dim=base_channel*2, window_size=self.window_size*(i+1), num_heads=num_heads[0], mlp_ratio=2.66,
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            )  for i in range(num_blocks[1]//2)
        ])

        self.emix_stage2 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel*4),
                SimpleSwinBlock(dim=base_channel*4, window_size=self.window_size*(i+1), num_heads=num_heads[1], mlp_ratio=2.66,
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            )  for i in range(num_blocks[2]//2)
        ])

        self.SCM1 = SCM(base_channel*2)
        self.SCM2 = SCM(base_channel*4)
        self.FAM1 = FAM(base_channel*2)
        self.FAM2 = FAM(base_channel*4)
    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)    
        x_4 = F.interpolate(x_2, scale_factor=0.5)  
        
        x_2 = self.SCM1(x_2)  
        x_4 = self.SCM2(x_4)

        x = self.cnn_stem(x)
        feature_list = []
        for i in range(len(self.ecnn_stage)):
            x = self.ecnn_stage[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[0](x)
        x = self.FAM1(x, x_2)
        for i in range(len(self.emix_stage1)):
            x = self.emix_stage1[i](x)
        feature_list.append(x)
        x = self.encoder_sampling[1](x)
        x = self.FAM2(x, x_4)
        for i in range(len(self.emix_stage2)):
            x = self.emix_stage2[i](x)
        
        feature_list.append(x)

        return feature_list



class Decoder(nn.Module):
    def __init__(self, base_channel, num_blocks=[4,4,4], num_heads=[8,4], mlp_ratio=2.66, window_size = 8, deep_supv=True):
        super(Decoder, self).__init__()
        self.window_size = window_size
        self.deep_supv = deep_supv
        if deep_supv:
            self.decoder_out_conv = nn.ModuleList([
            BasicConv(base_channel*4, 3, kernel_size=3, act=False, stride=1, bias=False),
            BasicConv(base_channel*2, 3, kernel_size=3, act=False, stride=1, bias=False),           
            ])
            
        
        self.output = BasicConv(base_channel, 3, kernel_size=3, act=False, stride=1, bias=False,norm=True)

        self.num_blocks_stage2 = num_blocks[1]
        self.num_blocks_stage3 = num_blocks[2]
        self.decoder_sampling = nn.ModuleList([   
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, act=True, stride=2, transpose=True, bias=False, norm=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, act=True, stride=2, transpose=True, bias=False, norm=True),
        ])

        self.decoder_compress_channel = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel*2, kernel_size=1, act=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, act=True, stride=1),
        ])

        self.dmix_stage2 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel*4),
                SimpleSwinBlock(dim=base_channel*4, window_size=self.window_size*(i+1), num_heads=num_heads[1], mlp_ratio=2.66,
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            )  for i in range(num_blocks[2]//2)
        ])
        self.dmix_stage1 = nn.ModuleList([
            nn.Sequential(
                VSSBlock(base_channel*2),
                SimpleSwinBlock(dim=base_channel*2, window_size=self.window_size*(i+1), num_heads=num_heads[1], mlp_ratio=2.66,
                    qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            )  for i in range(num_blocks[1]//2)
        ])

        self.dcnn_stage = nn.Sequential(
            *[
                ResidualDepthBlock(base_channel, base_channel) for i in range(num_blocks[0])
            ]
        )


    def forward(self, x0, enc_feature_list):
        if self.deep_supv:
            outputlist = []
            x_2 = F.interpolate(x0, scale_factor=0.5)  
            x_4 = F.interpolate(x_2, scale_factor=0.5)

        f1, f2, f3 = enc_feature_list
        x = f3
        for i in range(len(self.dmix_stage2)):
            x = self.dmix_stage2[i](x)
        if self.deep_supv:
            outputlist.append(self.decoder_out_conv[0](x)+x_4)

        x = self.decoder_sampling[0](x)
        x = torch.cat([x, f2], dim=1)
        x = self.decoder_compress_channel[0](x)

        for i in range(len(self.dmix_stage1)):
            x = self.dmix_stage1[i](x)
        if self.deep_supv:
            outputlist.append(self.decoder_out_conv[1](x)+x_2)

        x = self.decoder_sampling[1](x)
        x = torch.cat([x, f1], dim=1)
        x = self.decoder_compress_channel[1](x)

        for i in range(len(self.dcnn_stage)):
            x = self.dcnn_stage[i](x)
        x = self.output(x)
        if self.deep_supv:
            outputlist.append(x+x0)
            return outputlist
        return x + x0

    
class RestorMixer(nn.Module):
    def __init__(self, num_res=[4,4,4], base_channel=32, deep_supv=True):
        super(RestorMixer, self).__init__()
        
        self.encoder = Encoder(base_channel, num_blocks=num_res)
        self.decoder = Decoder(base_channel, num_blocks=num_res)

        self.apply(self._init_weights)

    def forward(self, x):
        enc_feature_list = self.encoder(x)

        pred = self.decoder(x, enc_feature_list)

        return pred
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def build_net():
    return RestorMixer()
