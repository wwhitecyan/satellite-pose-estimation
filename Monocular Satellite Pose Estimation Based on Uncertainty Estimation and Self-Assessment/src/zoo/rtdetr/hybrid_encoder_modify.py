"""by lyuwenyu
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from src.core import register
from .hybrid_encoder import HybridEncoder

__all__ = ["HybridEncoder_modify"]


@register
class HybridEncoder_modify(HybridEncoder):
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        proj_feats_0_resize = F.interpolate(
            proj_feats[0],
            size=(proj_feats[1].shape[2], proj_feats[1].shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        proj_feats_2_resize = F.interpolate(
            proj_feats[2],
            size=(proj_feats[1].shape[2], proj_feats[1].shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        # encoder
        # proj_feats_concat = torch.cat(
        #     (proj_feats[0], proj_feats[1], proj_feats[2]), dim=1
        # )
        # proj_feats_fusion = self.encoder_fusion_input(proj_feats_concat)
        # h, w = proj_feats_fusion.shape[2:]
        # src_flatten = proj_feats_fusion.flatten(2).permute(0, 2, 1)
        h, w = proj_feats_2_resize.shape[2:]

        src_flatten = proj_feats_2_resize.flatten(2).permute(0, 2, 1)

        if self.num_encoder_layers > 0:
            # self.use_encoder_idx: [2]
            for i, enc_ind in enumerate(self.use_encoder_idx):
                if self.training or self.eval_spatial_size is None:
                    # pos_embed shape 1,H*W, C
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                    # pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(
                    #     src_flatten.device
                    # )

                # memory shape bs,H*W,C
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # proj_feats
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, h, w)
                    .contiguous()
                )
                # print([x.is_contiguous() for x in proj_feats ])

        # proj_feats[-1] = self.conv_encoder(proj_feats[-1])
        proj_feats[0] = proj_feats_0_resize
        # proj_feats[2] = proj_feats_2_resize

        # # broadcasting and fusion
        # # inner_outs (last encoder output)shape : bs C 8 8
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):  # idx from 2 to 1
            feat_heigh = inner_outs[0]  # low resolution bs C 8 8
            feat_low = proj_feats[idx - 1]  # high resolution bs C 16 16
            # self.lateral_convs ConvNormLayer not change shape
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            # upsample_feat = F.interpolate(
            #     feat_heigh, scale_factor=2.0, mode="nearest"
            # )  # same shape with feat_low
            # inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
            #     torch.concat([upsample_feat, feat_low], dim=1)
            # )  # concat Channel
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                torch.concat([feat_heigh, feat_low], dim=1)
            )  # concat Channel
            inner_outs.insert(0, inner_out)  # to list

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):  # idx from 0 to 2
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            # downsample_feat = self.downsample_convs[idx](feat_low)
            # downsample_feat = F.interpolate(feat_low, scale_factor=0.5, mode="bicubic")
            # out = self.pan_blocks[idx](
            #     torch.concat([downsample_feat, feat_height], dim=1)
            # )
            out = self.pan_blocks[idx](torch.concat([feat_low, feat_height], dim=1))
            outs.append(out)

        # outs = [outs[0] + outs[1] + outs[2]]
        return outs
        # return proj_feats
