"""Guided-diffusion-style UNet building blocks for PMF experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import conv_nd, linear, normalization, timestep_embedding, zero_module
from guided_diffusion.unet import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    Upsample,
)


MIN_TIME_SCALE = 0.05


def convert_reconstruction_pair_to_velocity_pair(
    z: torch.Tensor,
    x_u_hat: torch.Tensor,
    x_v_hat: torch.Tensor,
    t: torch.Tensor | float | int,
    *,
    min_time_scale: float = MIN_TIME_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts one reconstruction pair to the PMF velocity pair parameterization."""
    if z.ndim != 4:
        raise ValueError("z must have shape (B, C, H, W).")
    if x_u_hat.shape != z.shape:
        raise ValueError("x_u_hat must have the same shape as z.")
    if x_v_hat.shape != z.shape:
        raise ValueError("x_v_hat must have the same shape as z.")

    if torch.is_tensor(t):
        t_tensor = t.to(device=z.device, dtype=z.dtype)
    else:
        t_tensor = torch.tensor(t, device=z.device, dtype=z.dtype)

    batch_size = z.shape[0]
    if t_tensor.ndim == 0:
        t_tensor = t_tensor.reshape(1).expand(batch_size)
    elif t_tensor.numel() == 1:
        t_tensor = t_tensor.reshape(1).expand(batch_size)
    elif t_tensor.numel() == batch_size:
        t_tensor = t_tensor.reshape(batch_size)
    else:
        raise ValueError("t must be a scalar or contain exactly one value per batch element.")

    time_scale = torch.clamp(t_tensor.reshape(batch_size, 1, 1, 1), min=min_time_scale)
    u = (z - x_u_hat) / time_scale
    v = (z - x_v_hat) / time_scale
    return u, v


def build_unet_channel_mult(
    image_size: int,
    channel_mult: str | Sequence[float | int] = "",
) -> tuple[float, ...]:
    """Builds the guided-diffusion channel multiplier tuple for one image size."""
    if isinstance(channel_mult, str):
        if channel_mult == "":
            if image_size == 512:
                return (0.5, 1, 1, 2, 2, 4, 4)
            if image_size == 256:
                return (1, 1, 2, 2, 4, 4)
            if image_size == 128:
                return (1, 1, 2, 3, 4)
            if image_size == 64:
                return (1, 2, 3, 4)
            raise ValueError(f"unsupported image size: {image_size}")
        return tuple(float(value) for value in channel_mult.split(","))
    return tuple(float(value) for value in channel_mult)


def build_attention_downsample_rates(
    image_size: int,
    attention_resolutions: str | Sequence[int],
) -> tuple[int, ...]:
    """Converts guided-diffusion attention resolutions to downsample rates."""
    if isinstance(attention_resolutions, str):
        if attention_resolutions == "":
            return ()
        resolutions = [int(value) for value in attention_resolutions.split(",")]
    else:
        resolutions = [int(value) for value in attention_resolutions]
    return tuple(image_size // resolution for resolution in resolutions)


class UNetPMFOutputHead(nn.Module):
    """Applies one zero-initialized image projection to UNet features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dims: int = 2,
    ) -> None:
        """Initializes the zero-initialized output head."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, in_channels, out_channels, 3, padding=1)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Projects one feature map to one image-like output tensor."""
        if features.ndim != 4:
            raise ValueError("features must have shape (B, C, H, W).")
        if features.shape[1] != self.in_channels:
            raise ValueError("features channel count does not match the output head.")
        return self.layers(features)


class _ScalarConditionProjector(nn.Module):
    """Projects one scalar PMF condition into the shared UNet embedding space."""

    def __init__(
        self,
        *,
        model_channels: int,
        emb_channels: int,
    ) -> None:
        """Initializes one zero-start scalar conditioning projector."""
        super().__init__()
        self.model_channels = model_channels
        self.emb_channels = emb_channels
        self.layers = nn.Sequential(
            linear(model_channels, emb_channels),
            nn.SiLU(),
            zero_module(linear(emb_channels, emb_channels)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Embeds one batch of scalar values to the UNet conditioning width."""
        if values.ndim != 1:
            raise ValueError("values must have shape (B,).")
        return self.layers(timestep_embedding(values, self.model_channels))


class UNetPMFConditioning(nn.Module):
    """Builds the shared PMF conditioning embedding for one UNet batch."""

    def __init__(
        self,
        *,
        model_channels: int,
        emb_channels: int,
        num_classes: int | None = None,
        use_aux_conditioning: bool = True,
        use_label_conditioning: bool = False,
    ) -> None:
        """Initializes the PMF-aligned conditioning module."""
        super().__init__()
        self.model_channels = model_channels
        self.emb_channels = emb_channels
        self.num_classes = num_classes
        self.use_aux_conditioning = use_aux_conditioning
        self.use_label_conditioning = use_label_conditioning and num_classes is not None

        self.h_embed = nn.Sequential(
            linear(model_channels, emb_channels),
            nn.SiLU(),
            linear(emb_channels, emb_channels),
        )
        self.omega_embed = _ScalarConditionProjector(
            model_channels=model_channels,
            emb_channels=emb_channels,
        )
        self.t_min_embed = _ScalarConditionProjector(
            model_channels=model_channels,
            emb_channels=emb_channels,
        )
        self.t_max_embed = _ScalarConditionProjector(
            model_channels=model_channels,
            emb_channels=emb_channels,
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, emb_channels)
            self.label_proj = zero_module(linear(emb_channels, emb_channels))
        else:
            self.label_emb = None
            self.label_proj = None

    def forward(
        self,
        h: torch.Tensor | float | int,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        omega: torch.Tensor | float | int | None = None,
        t_min: torch.Tensor | float | int | None = None,
        t_max: torch.Tensor | float | int | None = None,
        labels: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Builds one PMF conditioning embedding batch."""
        h_tensor = self._normalize_scalar(
            name="h",
            value=h,
            batch_size=batch_size,
            default=None,
            dtype=dtype,
            device=device,
        )
        batch_size = h_tensor.shape[0]
        device = h_tensor.device
        emb = self.h_embed(timestep_embedding(h_tensor, self.model_channels))

        if self.use_aux_conditioning:
            omega_tensor = self._normalize_scalar(
                name="omega",
                value=omega,
                batch_size=batch_size,
                default=1.0,
                dtype=h_tensor.dtype,
                device=device,
            )
            t_min_tensor = self._normalize_scalar(
                name="t_min",
                value=t_min,
                batch_size=batch_size,
                default=0.0,
                dtype=h_tensor.dtype,
                device=device,
            )
            t_max_tensor = self._normalize_scalar(
                name="t_max",
                value=t_max,
                batch_size=batch_size,
                default=1.0,
                dtype=h_tensor.dtype,
                device=device,
            )
            emb = emb + self.omega_embed(omega_tensor)
            emb = emb + self.t_min_embed(t_min_tensor)
            emb = emb + self.t_max_embed(t_max_tensor)

        if self.use_label_conditioning:
            label_tensor = self._normalize_labels(
                labels=labels,
                batch_size=batch_size,
                device=device,
            )
            assert self.label_emb is not None
            assert self.label_proj is not None
            emb = emb + self.label_proj(self.label_emb(label_tensor))

        return emb

    @staticmethod
    def _normalize_scalar(
        *,
        name: str,
        value: torch.Tensor | float | int | None,
        batch_size: int | None,
        default: float | None,
        dtype: torch.dtype,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Normalizes one scalar condition to shape ``(B,)``."""
        if value is None:
            if default is None:
                raise ValueError(f"{name} must be provided.")
            if batch_size is None:
                raise ValueError(f"{name} batch size is required when using a default.")
            tensor = torch.full((batch_size,), default, dtype=dtype, device=device)
            return tensor

        if torch.is_tensor(value):
            tensor = value.to(device=device, dtype=dtype)
        else:
            tensor = torch.tensor(value, device=device, dtype=dtype)

        if tensor.ndim == 0:
            if batch_size is None:
                return tensor.reshape(1)
            return tensor.reshape(1).expand(batch_size)
        if tensor.numel() == 1 and batch_size is not None:
            return tensor.reshape(1).expand(batch_size)
        if batch_size is None:
            return tensor.reshape(-1)
        if tensor.numel() != batch_size:
            raise ValueError(
                f"{name} must be a scalar or contain exactly one value per batch element."
            )
        return tensor.reshape(batch_size)

    def _normalize_labels(
        self,
        *,
        labels: torch.Tensor | int | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalizes one optional label batch."""
        if labels is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if torch.is_tensor(labels):
            label_tensor = labels.to(device=device, dtype=torch.long)
        else:
            label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
        if label_tensor.ndim == 0:
            return label_tensor.reshape(1).expand(batch_size)
        if label_tensor.numel() == 1:
            return label_tensor.reshape(1).expand(batch_size)
        if label_tensor.numel() != batch_size:
            raise ValueError("labels must be a scalar or contain one label per batch item.")
        return label_tensor.reshape(batch_size)


class UNetPMFBackbone(nn.Module):
    """Guided-diffusion-style UNet torso driven by external conditioning embeddings."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        emb_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: str | Sequence[int] = "16",
        dropout: float = 0.0,
        channel_mult: str | Sequence[float | int] = "",
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ) -> None:
        """Initializes the UNet torso without task-specific output parameterization."""
        super().__init__()
        del use_new_attention_order

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.emb_channels = emb_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.channel_mult = build_unet_channel_mult(image_size, channel_mult)
        self.attention_downsample_rates = build_attention_downsample_rates(
            image_size,
            attention_resolutions,
        )

        ch = input_ch = int(self.channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        input_block_channels = [ch]
        ds = 1

        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        emb_channels,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_downsample_rates:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            emb_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_channels.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                emb_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock(
                ch,
                emb_channels,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for block_index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        ch + skip_channels,
                        emb_channels,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_downsample_rates:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                        )
                    )
                if level and block_index == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            emb_channels,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.output_channels = ch
        self.input_projection_channels = input_ch

    def convert_to_fp16(self) -> None:
        """Converts the UNet torso blocks to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """Converts the UNet torso blocks back to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the UNet torso to one image batch and one conditioning batch."""
        if x.ndim != 4:
            raise ValueError("x must have shape (B, C, H, W).")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("x spatial size must match image_size.")
        if emb.ndim != 2:
            raise ValueError("emb must have shape (B, E).")
        if emb.shape[0] != x.shape[0]:
            raise ValueError("emb batch size must match x batch size.")
        if emb.shape[1] != self.emb_channels:
            raise ValueError("emb feature size must match emb_channels.")

        hidden_states = []
        h = x.type(self.dtype)
        emb = emb.to(device=x.device, dtype=h.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hidden_states.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hidden_states.pop()], dim=1)
            h = module(h, emb)
        return h.type(x.dtype)


class UNetPMFModel(nn.Module):
    """Convenience wrapper combining the PMF UNet torso with one output head."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: str | Sequence[int] = "16",
        dropout: float = 0.0,
        channel_mult: str | Sequence[float | int] = "",
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ) -> None:
        """Initializes one complete UNet PMF model with one zero-init image head."""
        super().__init__()
        self.backbone = UNetPMFBackbone(
            image_size=image_size,
            in_channels=in_channels,
            emb_channels=emb_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
        self.out = UNetPMFOutputHead(
            in_channels=self.backbone.output_channels,
            out_channels=out_channels,
            dims=dims,
        )

    def convert_to_fp16(self) -> None:
        """Converts the UNet torso blocks to float16."""
        self.backbone.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """Converts the UNet torso blocks back to float32."""
        self.backbone.convert_to_fp32()

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the PMF UNet model to one image batch."""
        return self.out(self.backbone(x, emb))


class UNetPMFVelocityPairModel(nn.Module):
    """Combines the PMF UNet torso with two zero-init reconstruction heads."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        emb_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: str | Sequence[int] = "16",
        dropout: float = 0.0,
        channel_mult: str | Sequence[float | int] = "",
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        min_time_scale: float = MIN_TIME_SCALE,
    ) -> None:
        """Initializes one dual-head PMF UNet with external time-to-velocity conversion."""
        super().__init__()
        self.min_time_scale = min_time_scale
        self.backbone = UNetPMFBackbone(
            image_size=image_size,
            in_channels=in_channels,
            emb_channels=emb_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
        self.u_head = UNetPMFOutputHead(
            in_channels=self.backbone.output_channels,
            out_channels=in_channels,
            dims=dims,
        )
        self.v_head = UNetPMFOutputHead(
            in_channels=self.backbone.output_channels,
            out_channels=in_channels,
            dims=dims,
        )

    def convert_to_fp16(self) -> None:
        """Converts the UNet torso blocks to float16."""
        self.backbone.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """Converts the UNet torso blocks back to float32."""
        self.backbone.convert_to_fp32()

    def predict_reconstruction_pair(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the image-like pair ``(x_u_hat, x_v_hat)`` from shared features."""
        features = self.backbone(z, emb)
        return self.u_head(features), self.v_head(features)

    def forward(
        self,
        z: torch.Tensor,
        emb: torch.Tensor,
        *,
        t: torch.Tensor | float | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the PMF velocity pair produced from the reconstruction pair."""
        x_u_hat, x_v_hat = self.predict_reconstruction_pair(z, emb)
        return convert_reconstruction_pair_to_velocity_pair(
            z,
            x_u_hat,
            x_v_hat,
            t,
            min_time_scale=self.min_time_scale,
        )


class ConditionedUNetPMFModel(nn.Module):
    """Combines PMF conditioning, the UNet torso, and one zero-init image head."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: str | Sequence[int] = "16",
        dropout: float = 0.0,
        channel_mult: str | Sequence[float | int] = "",
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        num_classes: int | None = None,
        use_aux_conditioning: bool = True,
        use_label_conditioning: bool = False,
    ) -> None:
        """Initializes one single-head PMF UNet with PMF-style conditioning."""
        super().__init__()
        self.conditioning = UNetPMFConditioning(
            model_channels=model_channels,
            emb_channels=emb_channels,
            num_classes=num_classes,
            use_aux_conditioning=use_aux_conditioning,
            use_label_conditioning=use_label_conditioning,
        )
        self.model = UNetPMFModel(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            emb_channels=emb_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )

    def convert_to_fp16(self) -> None:
        """Converts the UNet torso blocks to float16."""
        self.model.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """Converts the UNet torso blocks back to float32."""
        self.model.convert_to_fp32()

    def forward(
        self,
        x: torch.Tensor,
        *,
        h: torch.Tensor | float | int,
        omega: torch.Tensor | float | int | None = None,
        t_min: torch.Tensor | float | int | None = None,
        t_max: torch.Tensor | float | int | None = None,
        labels: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Applies the conditioned PMF UNet model to one image batch."""
        emb = self.conditioning(
            h,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        return self.model(x, emb)


class ConditionedUNetPMFVelocityPairModel(nn.Module):
    """Combines PMF conditioning, the UNet torso, and dual PMF velocity heads."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        emb_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: str | Sequence[int] = "16",
        dropout: float = 0.0,
        channel_mult: str | Sequence[float | int] = "",
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        num_classes: int | None = None,
        use_aux_conditioning: bool = True,
        use_label_conditioning: bool = False,
        min_time_scale: float = MIN_TIME_SCALE,
    ) -> None:
        """Initializes one conditioned dual-head PMF UNet."""
        super().__init__()
        self.conditioning = UNetPMFConditioning(
            model_channels=model_channels,
            emb_channels=emb_channels,
            num_classes=num_classes,
            use_aux_conditioning=use_aux_conditioning,
            use_label_conditioning=use_label_conditioning,
        )
        self.model = UNetPMFVelocityPairModel(
            image_size=image_size,
            in_channels=in_channels,
            emb_channels=emb_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            min_time_scale=min_time_scale,
        )

    def convert_to_fp16(self) -> None:
        """Converts the UNet torso blocks to float16."""
        self.model.convert_to_fp16()

    def convert_to_fp32(self) -> None:
        """Converts the UNet torso blocks back to float32."""
        self.model.convert_to_fp32()

    def predict_reconstruction_pair(
        self,
        z: torch.Tensor,
        *,
        h: torch.Tensor | float | int,
        omega: torch.Tensor | float | int | None = None,
        t_min: torch.Tensor | float | int | None = None,
        t_max: torch.Tensor | float | int | None = None,
        labels: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the PMF reconstruction pair before velocity conversion."""
        emb = self.conditioning(
            h,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        return self.model.predict_reconstruction_pair(z, emb)

    def forward(
        self,
        z: torch.Tensor,
        *,
        t: torch.Tensor | float | int,
        h: torch.Tensor | float | int,
        omega: torch.Tensor | float | int | None = None,
        t_min: torch.Tensor | float | int | None = None,
        t_max: torch.Tensor | float | int | None = None,
        labels: torch.Tensor | int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the conditioned PMF UNet and returns the PMF velocity pair."""
        emb = self.conditioning(
            h,
            batch_size=z.shape[0],
            device=z.device,
            dtype=z.dtype,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            labels=labels,
        )
        return self.model(z, emb, t=t)


__all__ = [
    "ConditionedUNetPMFModel",
    "ConditionedUNetPMFVelocityPairModel",
    "UNetPMFConditioning",
    "UNetPMFBackbone",
    "UNetPMFModel",
    "UNetPMFOutputHead",
    "UNetPMFVelocityPairModel",
    "build_attention_downsample_rates",
    "build_unet_channel_mult",
    "convert_reconstruction_pair_to_velocity_pair",
]
