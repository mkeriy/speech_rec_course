from typing import Optional, Union

from omegaconf import DictConfig
import hydra
import torch
from torch import nn

from src.submodules.subsampling import StackingSubsampling
from src.submodules.positional_encoding import PositionalEncoding


class ConvolutionalSpatialGatingUnit(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Spatial Gating Unit (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size in DepthWise Conv
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """

        super().__init__()
        # TODO: LayerNorm
        self.norm = nn.LayerNorm(size // 2)

        # TODO: DepthWise Conv
        self.conv = nn.Conv1d(
            size // 2,
            size // 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=size // 2,
        )

        if use_linear_after_conv:
            self.linear = nn.Linear(size // 2, size // 2)
        else:
            self.linear = None
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        Inputs:
            x: B x T x C
        Outputs:
            out: B x T x C
        """
        e_1, e_2 = x.chunk(2, dim=-1)

        e_2 = self.conv(self.norm(e_2).transpose(1, 2)).transpose(1, 2)

        if self.linear:
            e_2 = self.linear(e_2)

        e = e_1 * e_2
        return e


class ConvolutionalGatingMLP(torch.nn.Module):
    def __init__(
        self,
        size: int,
        kernel_size: int,
        expansion_factor: int = 6,
        dropout: float = 0.0,
        use_linear_after_conv: bool = False,
    ):
        """
        Convolutional Gating MLP (https://arxiv.org/pdf/2207.02971)
        Args:
            size: int - Input embedding dim
            kernel_size: int - Kernel size for DepthWise Conv in ConvolutionalSpatialGatingUnit
            expansion_factor: int - Dim expansion factor for ConvolutionalSpatialGatingUnit
            dropout: float - Dropout rate
            use_linear_after_conv: bool - Whether to use linear layer after convolution
        """
        super().__init__()

        # TODO: First Channel Projection with GeLU Activation
        self.channel_proj1 = nn.Sequential(## Missed LayerNorm, why?
            nn.Linear(size, size * expansion_factor),
            nn.GELU(),
        )
        # TODO: Convlutional Spatial Gating Unit
        self.csgu = ConvolutionalSpatialGatingUnit(
            size=size * expansion_factor,
            kernel_size=kernel_size,
            dropout=dropout,
            use_linear_after_conv=use_linear_after_conv,
        )

        # TODO: Second Channel Projection with GeLU Activation
        self.channel_proj2 = nn.Sequential(
            nn.Linear(size * expansion_factor // 2, size), nn.Dropout(p=dropout)
        )

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C

        """
        e_local = self.channel_proj1(features)
        e_local = self.csgu(e_local)

        e_local = self.channel_proj2(e_local)
        return e_local


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        """
        Standard FeedForward layer from Transformer block,
        consisting of a compression and decompression projection
        with an activation function.
        Args:
            input_dim: int - Input embedding dim
            hidden_dim: int - Hidden dim
            dropout: float - Dropout rate
            activation: torch.nn.Module - Activation function
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation

    def forward(self, features: torch.Tensor):
        """
        Inputs:
            features: B x T x C
        Outputs:
            out: B x T x C
        """
        out = self.linear1(features)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        return out


class EBranchformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        size: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 4,
        dropout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        """
        E-Bbranchformer Layer (https://arxiv.org/pdf/2210.00077)
        Args:
            size: int - Embedding dim
            attn_config: DictConfig or dict - Config for MultiheadAttention
            cgmlp_config: DictConfig or dict - Config for ConvolutionalGatingMLP
            ffn_expansion_factor: int - Expansion factor for FeedForward
            dropout: float - Dropout rate
            merge_conv_kernel: int - Kernel size for merging module
        """

        super().__init__()

        # MultiheadAttention from torch.nn
        self.attn = nn.MultiheadAttention(**attn_config)

        # ConvolutionalGatingMLP module
        self.cgmlp = ConvolutionalGatingMLP(**cgmlp_config)

        # First and Second FeedForward modules
        self.feed_forward1 = FeedForward(
            size, size * ffn_expansion_factor, dropout=dropout
        )
        self.feed_forward2 = FeedForward(
            size, size * ffn_expansion_factor, dropout=dropout
        )

        # Normalization modules
        self.norm_ffn1 = nn.LayerNorm(size)
        self.norm_ffn2 = nn.LayerNorm(size)
        self.norm_mha = nn.LayerNorm(size)
        self.norm_mlp = nn.LayerNorm(size)
        self.norm_final = nn.LayerNorm(size)

        self.dropout = nn.Dropout(p=dropout)

        # DepthWise Convolution and Linear projection for merging module
        self.depthwise_conv_fusion = nn.Conv1d(
            2 * size,
            2 * size,
            kernel_size=merge_conv_kernel,
            padding=merge_conv_kernel // 2,
            groups=2 * size,
        )
        self.merge_proj = nn.Linear(2 * size, size)

    def forward(
        self,
        features: torch.Tensor,
        features_length: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ):
        """
        Inputs:
            features: B x T x C
            features_length: B
            pos_emb: B x T x C - Optional
        Outputs:
            out: B x T x C
        """
        e = features + 0.5 * self.feed_forward1(self.norm_ffn1(features))
        e_global, _ = self.attn(
            query=self.norm_mha(e),
            key=self.norm_mha(e),
            value=self.norm_mha(e),
            need_weights=False,
            key_padding_mask=pos_emb,
        )
        e_local = self.cgmlp(self.norm_mlp(e))

        e_merged = torch.cat([e_local, e_global], dim=-1)
        e_conv = self.depthwise_conv_fusion(e_merged.transpose(1, 2)).transpose(1, 2)
        e_merged = self.dropout(self.merge_proj(e_merged + e_conv))

        e = e + e_merged

        e = e + 0.5 * self.feed_forward2(self.norm_ffn2(e))
        e = self.norm_final(e)
        return e


class EBranchformerEncoder(torch.nn.Module):
    def __init__(
        self,
        subsampling_stride: int,
        features_num: int,
        d_model: int,
        layers_num: int,
        attn_config: Union[DictConfig, dict],
        cgmlp_config: Union[DictConfig, dict],
        ffn_expansion_factor: int = 2,
        dropout: float = 0.0,
        merge_conv_kernel: int = 3,
    ):
        super().__init__()

        self.subsampling = StackingSubsampling(
            stride=subsampling_stride, feat_in=features_num, feat_out=d_model
        )

        self.pos_embedding = PositionalEncoding(d_model, dropout)

        self.layers = torch.nn.ModuleList()
        for _ in range(layers_num):

            layer = EBranchformerEncoderLayer(
                size=d_model,
                attn_config=attn_config,
                cgmlp_config=cgmlp_config,
                ffn_expansion_factor=ffn_expansion_factor,
                dropout=dropout,
                merge_conv_kernel=merge_conv_kernel,
            )
            self.layers.append(layer)

    def forward(self, features: torch.Tensor, features_length: torch.Tensor):
        features = features.transpose(1, 2)  # B x D x T -> B x T x D
        features, features_length = self.subsampling(features, features_length)
        features = self.pos_embedding(features)
        for layer in self.layers:
            features = layer(features, features_length)

        return features, features_length
