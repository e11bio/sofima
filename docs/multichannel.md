# Multichannel Image Support

SOFIMA supports multichannel (e.g., multi-fluorescence, RGB) microscopy data.
This document describes how multichannel data flows through the pipeline and
how to configure each stage.

## Design Philosophy

Optical flow estimation computes a **spatial transformation** between two images.
SOFIMA supports two modes for handling multichannel data:

1. **Single-channel selection** (`channel=int`): Use one channel for alignment,
   then apply the result to all channels during warping. Useful when one channel
   has clearly superior contrast.

2. **Multi-channel averaging** (`channel=[int, ...]`): Compute cross-correlations
   independently on each specified channel and average them before peak detection.
   This leverages information from all channels simultaneously, improving
   alignment robustness when no single channel is dominant.

Both approaches apply the resulting transformation to all channels uniformly
during warping.

## Data Layout

Throughout SOFIMA, image data follows these conventions:

| Dimensionality | Layout | Description |
|---|---|---|
| 2D single-channel | `(y, x)` | Spatial image |
| 2D multichannel | `(c, y, x)` | Channel-first |
| 3D single-channel | `(z, y, x)` | Volumetric |
| 3D multichannel | `(c, z, y, x)` | Channel-first volumetric |
| Warping input | `(n, z, y, x)` | `n` channels warped with shared coordinate map |

## Usage Guide

### Flow Field Estimation

```python
from sofima import flow_field

calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()

# Multichannel images: shape (c, y, x) or (c, z, y, x)
pre_image = ...  # (3, 1024, 1024) for 3-channel 2D
post_image = ...

# Option 1: Single channel (e.g., DAPI)
flow = calculator.flow_field(
    pre_image, post_image,
    patch_size=80, step=40,
    channel=0  # Use only channel 0
)

# Option 2: Multi-channel averaging (use all channels)
flow = calculator.flow_field(
    pre_image, post_image,
    patch_size=80, step=40,
    channel=[0, 1, 2]  # Average xcorr across all 3 channels
)
```

### Rigid Stitching

```python
from sofima import stitch_rigid

# tile_map values can be (c, y, x) multichannel arrays
tile_map = {
    (0, 0): tile_00,  # shape: (3, 2048, 2048)
    (1, 0): tile_10,
    ...
}

# Single channel alignment
conn_x, conn_y = stitch_rigid.compute_coarse_offsets(
    yx_shape=(2, 3),
    tile_map=tile_map,
    channel=1  # Select one channel
)

# Multi-channel alignment (average xcorr across channels)
conn_x, conn_y = stitch_rigid.compute_coarse_offsets(
    yx_shape=(2, 3),
    tile_map=tile_map,
    channel=[0, 1, 2]  # Use all channels
)
```

### Elastic Stitching

```python
from sofima import stitch_elastic

# 2D tiles with multi-channel averaging
fine_x, offsets_x = stitch_elastic.compute_flow_map(
    tile_map=tile_map,
    offset_map=conn_x,
    axis=0,
    channel=[0, 1]  # Average xcorr over channels 0 and 1
)

# 3D tiles: tile_map values have shape (c, z, y, x)
fine_x_3d, offsets_x_3d = stitch_elastic.compute_flow_map3d(
    tile_map=tile_map_3d,
    tile_shape=(256, 256, 64),
    offset_map=offset_map,
    axis=0,
    channel=[0, 1, 2]  # Use all channels
)
```

### Warping Multichannel Data

```python
from sofima import warp

# image shape: (n_channels, z, y, x)
multichannel_volume = ...  # (3, 100, 1024, 1024)

# warp_subvolume handles all channels automatically
warped = warp.warp_subvolume(
    multichannel_volume,
    image_box, coord_map, map_box, stride, out_box
)
# warped shape: (3, z_out, y_out, x_out)
```

### Processor Pipeline

```python
from sofima.processor import flow

# Single channel
config = flow.EstimateFlow.Config(
    patch_size=160,
    stride=40,
    z_stride=1,
    fixed_current=False,
    mask_configs=None,
    mask_only_for_patch_selection=False,
    selection_mask_configs=None,
    batch_size=256,
    channel=0  # Use channel 0
)

# Multi-channel averaging
config = flow.EstimateFlow.Config(
    ...,
    channel=[0, 1, 2]  # Average xcorr across all channels
)
```

## Default Behavior (`channel=None`)

When `channel=None` (the default), all functions behave identically to their
original (pre-multichannel) implementation. There are no breaking API changes:

- `channel=None` in `flow_field()`: images used as-is (assumed spatial-only).
- `channel=None` in `compute_coarse_offsets()`: tiles used as-is (assumed 2D
  `(y, x)`, identical to prior behavior).
- `channel=None` in `compute_flow_map()`: tiles used as-is (assumed 2D `(y, x)`,
  identical to prior behavior).
- `channel=None` in `compute_flow_map3d()`: tiles assumed to have shape
  `[1, z, y, x]` and the leading dimension is squeezed (identical to prior
  behavior).
- `channel=None` in `EstimateFlow.Config`: uses channel 0 (identical to prior
  behavior).

To enable multichannel support, explicitly specify the `channel` parameter.

## Best Practices

1. **Multi-channel averaging is preferred** when multiple channels have useful
   signal. Averaging cross-correlations across channels provides more robust
   alignment than any single channel alone.

2. **Fall back to single-channel** when one channel is clearly dominant in
   contrast/features, or when other channels are noisy or uninformative.

3. **Consistent channel selection**: Use the same channel parameter across all
   pipeline stages (rigid stitching, elastic stitching, section alignment).

4. **Memory considerations**: For multi-channel averaging, all selected channels
   are loaded into GPU memory. For large images, consider selecting a subset of
   channels rather than all channels if memory is limited.
