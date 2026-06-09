# Multichannel Image Support

SOFIMA supports multichannel (e.g., multi-fluorescence, RGB) microscopy data.
This document describes how multichannel data flows through the pipeline and
how to configure each stage.

## Design Philosophy

Optical flow estimation computes a **spatial transformation** between two images.
This transformation is computed from a single intensity channel (or a derived
scalar representation) and then applied to all channels uniformly during warping.

This approach:
- Avoids redundant computation (same displacement applies to all channels)
- Allows selecting the best-contrast channel for alignment
- Preserves all channel data through the warping step

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

# Compute flow using channel 0 (e.g., DAPI)
flow = calculator.flow_field(
    pre_image, post_image,
    patch_size=80, step=40,
    channel=0  # Select channel for flow computation
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

# Use channel 1 for alignment
conn_x, conn_y = stitch_rigid.compute_coarse_offsets(
    yx_shape=(2, 3),
    tile_map=tile_map,
    channel=1  # Select alignment channel
)
```

### Elastic Stitching

```python
from sofima import stitch_elastic

# 2D tiles with channel selection
fine_x, offsets_x = stitch_elastic.compute_flow_map(
    tile_map=tile_map,
    offset_map=conn_x,
    axis=0,
    channel=0  # Select channel for fine alignment
)

# 3D tiles: tile_map values have shape (c, z, y, x)
fine_x_3d, offsets_x_3d = stitch_elastic.compute_flow_map3d(
    tile_map=tile_map_3d,
    tile_shape=(256, 256, 64),
    offset_map=offset_map,
    axis=0,
    channel=0  # Select channel (defaults to 0)
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

config = flow.EstimateFlow.Config(
    patch_size=160,
    stride=40,
    z_stride=1,
    fixed_current=False,
    mask_configs=None,
    mask_only_for_patch_selection=False,
    selection_mask_configs=None,
    batch_size=256,
    channel=0  # Use channel 0 from multichannel input volume
)
```

## Backward Compatibility

All multichannel parameters default to `None` (or `0` for `compute_flow_map3d`),
preserving full backward compatibility with existing single-channel workflows:

- `channel=None` in `flow_field()`: images used as-is (spatial-only arrays)
- `channel=None` in `compute_coarse_offsets()`: tiles used as-is (2D arrays)
- `channel=None` in `compute_flow_map()`: tiles used as-is (2D arrays)
- `channel=None` (defaulting to 0) in `compute_flow_map3d()`: extracts first
  channel, matching the prior `squeeze(axis=0)` behavior

## Best Practices

1. **Choose the best channel**: Select the channel with highest contrast and
   most features for flow estimation. Nuclear stains (e.g., DAPI) often work
   well for alignment.

2. **Consistent channel selection**: Use the same channel parameter across all
   pipeline stages (rigid stitching, elastic stitching, section alignment).

3. **Memory considerations**: Only the selected channel is loaded into GPU
   memory for flow computation. Multichannel warping processes channels
   sequentially to limit memory usage.
