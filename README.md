# SOFIMA

SOFIMA (Scalable Optical Flow-based Image Montaging and Alignment) is a tool
for stitching, aligning and warping large 2d, 3d and 4d microscopy datasets.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10534541.svg)](https://zenodo.org/records/10534541)

Please cite as https://zenodo.org/records/10534541

This is not an officially supported Google product.

# Installation

SOFIMA is implemented purely in Python, and does not require a build step. To
install it directly from the repository, run:

```shell
  pip install git+https://github.com/google-research/sofima
```

# Overview

SOFIMA uses optical flow regularized with an elastic mesh to establish
maps between data in different coordinate systems. Both the [flow estimator](flow_field.py)
as well as the [mesh solver](mesh.py) are implemented in [JAX](https://github.com/google/jax)
and will automatically take advantage of GPU acceleration if the hardware if available.

A core data structure used throughout the project is a *coordinate map* stored
as a dense array of relative offsets (see the module docstring in [map_utils.py](map_utils.py)
for details). Among other uses, this is the representation of the estimated flow fields
and the mesh node positions.

## Multichannel Support

SOFIMA supports multichannel image data throughout the pipeline. The typical
workflow for multichannel data is:

1. **Flow estimation**: Compute the flow field using a single channel from the
   multichannel input. Use the `channel` parameter in
   `JAXMaskedXCorrWithStatsCalculator.flow_field()`, `stitch_rigid.compute_coarse_offsets()`,
   and `stitch_elastic.compute_flow_map()` / `compute_flow_map3d()` to select
   which channel to use for alignment.

2. **Warping**: Apply the computed flow/coordinate map to all channels
   simultaneously. The `warp_subvolume()` function accepts `[n, z, y, x]` data
   where `n` is the number of channels, and warps each channel independently
   using the same coordinate map.

This design reflects the fact that optical flow estimation fundamentally
computes a spatial transformation, which is then applied uniformly to all
channels of the source data.

# Example usage

 * [electron microscopy tile stitching](https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_stitching.ipynb)
 * [electron microscopy section alignment](https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/em_alignment.ipynb)
 * [LICONN 3d tile stitching](https://colab.research.google.com/github/google-research/sofima/blob/main/notebooks/liconn_inplane_stitching.ipynb)

# Citation

If you use this software in your research, please cite it using the following metadata:

> Januszewski, M., Blakely, T., & Lueckmann, J.-M. (2024). SOFIMA: Scalable Optical Flow-based Image Montaging and Alignment. Zenodo. https://doi.org/10.5281/zenodo.10534541

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
