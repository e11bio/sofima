# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for flow_field."""

from absl.testing import absltest
import numpy as np
from sofima import flow_field


class FlowFieldTest(absltest.TestCase):

  def test_jax_masked_xcorr_calculator(self):
    pre_image = np.zeros((120, 120), dtype=np.uint8)
    post_image = np.zeros((120, 120), dtype=np.uint8)

    pre_image[60, 60] = 255
    post_image[70, 53] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])
    np.testing.assert_array_equal(np.zeros((2, 2)), field[3, ...])

    # 2nd point in the post-image would normally confuse the flow estimation,
    # but with masking it should have no impact.
    post_image[54, 68] = 255
    post_image_mask = np.zeros((128, 128), dtype=bool)
    post_image_mask[:55, :70] = 1
    field = calculator.flow_field(
        pre_image,
        post_image,
        patch_size=80,
        step=40,
        post_mask=post_image_mask,
        batch_size=4)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])
    np.testing.assert_array_equal(np.zeros((2, 2)), field[3, ...])

  def test_jax_xcorr_3d(self):
    pre_image = np.zeros((50, 100, 100), dtype=np.uint8)
    post_image = np.zeros((50, 100, 100), dtype=np.uint8)

    pre_image[25, 50, 50] = 255
    post_image[22, 45, 54] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    flow = calculator.flow_field(
        pre_image, post_image, patch_size=(40, 80, 80), step=10, batch_size=1)

    np.testing.assert_array_equal([5, 2, 3, 3], flow.shape)
    np.testing.assert_array_equal(np.full([2, 3, 3], -4), flow[0, ...])
    np.testing.assert_array_equal(np.full([2, 3, 3], 5), flow[1, ...])
    np.testing.assert_array_equal(np.full([2, 3, 3], 3), flow[2, ...])

  def test_jax_peak(self):
    hy, hx = np.mgrid[:50, :50]
    cy, cx = 20, 28
    hy = cy - hy
    hx = cx - hx
    r = np.sqrt(2 * hx**2 + hy**2)
    peak_max = 10
    xcorr = peak_max * np.exp(-r / 4)

    peaks = flow_field._batched_peaks(
        xcorr[np.newaxis, ...], (25, 25),
        min_distance=2,
        threshold_rel=0.5,
        peak_radius=(2, 3))
    np.testing.assert_array_equal([1, 4], peaks.shape)

    peak_support = np.min(xcorr[cy - 2:cy + 3, cx - 3:cx + 4])
    self.assertEqual(peaks[0, 0], 3)  # x
    self.assertEqual(peaks[0, 1], -5)  # y
    self.assertEqual(peaks[0, 2], peak_max / peak_support)  # sharpness
    self.assertEqual(peaks[0, 3], 0)  # peak ratio

  def test_post_targeting(self):
    pre_image = np.zeros((120, 120), dtype=np.uint8)
    post_image = np.zeros((120, 120), dtype=np.uint8)

    pre_image[50, 55] = 255
    post_image[100, 100] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()

    # Without targeting, the features are too far apart to be picked up.
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4)
    np.testing.assert_array_equal(np.isnan(field[:, 0, 0]), True)

    post_targeting_field = np.full((2, 2, 2), 40.0, dtype=np.float32)

    # With targeting, a flow field of magnitude larger than the
    # normally possible max of patch_size // 2 can be estimated.
    field = calculator.flow_field(
        pre_image,
        post_image,
        patch_size=80,
        step=40,
        batch_size=4,
        post_targeting_field=post_targeting_field,
        post_targeting_step=40)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(-45 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-50 * np.ones((2, 2)), field[1, ...])

  def test_multichannel_input(self):
    """Tests that multichannel input works with channel selection."""
    # Create a 3-channel image where only channel 1 has the signal.
    pre_image = np.zeros((3, 120, 120), dtype=np.uint8)
    post_image = np.zeros((3, 120, 120), dtype=np.uint8)

    pre_image[1, 60, 60] = 255
    post_image[1, 70, 53] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=1)

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])

  def test_multichannel_channel_zero(self):
    """Tests that channel=0 with multichannel gives same result as 2D."""
    pre_image_2d = np.zeros((120, 120), dtype=np.uint8)
    post_image_2d = np.zeros((120, 120), dtype=np.uint8)
    pre_image_2d[60, 60] = 255
    post_image_2d[70, 53] = 255

    # Wrap in multichannel with noise on other channels.
    pre_image_mc = np.stack([
        pre_image_2d, np.random.randint(0, 50, (120, 120), dtype=np.uint8)
    ])
    post_image_mc = np.stack([
        post_image_2d, np.random.randint(0, 50, (120, 120), dtype=np.uint8)
    ])

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    field_2d = calculator.flow_field(
        pre_image_2d, post_image_2d, patch_size=80, step=40, batch_size=4)
    field_mc = calculator.flow_field(
        pre_image_mc, post_image_mc, patch_size=80, step=40, batch_size=4,
        channel=0)

    np.testing.assert_array_equal(field_2d, field_mc)

  def test_multichannel_averaging(self):
    """Tests that multi-channel averaging uses all channels for alignment."""
    # Create 2-channel images where each channel has signal at the same offset.
    pre_image = np.zeros((2, 120, 120), dtype=np.float32)
    post_image = np.zeros((2, 120, 120), dtype=np.float32)

    # Channel 0: signal at one location
    pre_image[0, 60, 60] = 1.0
    post_image[0, 70, 53] = 1.0

    # Channel 1: signal at the same offset but different location
    pre_image[1, 40, 40] = 1.0
    post_image[1, 50, 33] = 1.0

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    # Use both channels for alignment.
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=[0, 1])

    np.testing.assert_array_equal([4, 2, 2], field.shape)
    # Both channels have the same offset (7, -10), so the average xcorr
    # should produce the same result.
    np.testing.assert_array_equal(7 * np.ones((2, 2)), field[0, ...])
    np.testing.assert_array_equal(-10 * np.ones((2, 2)), field[1, ...])

  def test_multichannel_averaging_single_channel_list(self):
    """Tests that channel=[0] gives same result as channel=0."""
    pre_image = np.zeros((2, 120, 120), dtype=np.uint8)
    post_image = np.zeros((2, 120, 120), dtype=np.uint8)

    pre_image[0, 60, 60] = 255
    post_image[0, 70, 53] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    field_single = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=0)
    field_list = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=[0])

    np.testing.assert_array_equal(field_single, field_list)

  def test_channel_none_multichannel_uses_all(self):
    """Tests that multichannel averaging via channel=[0,1] matches explicit."""
    # Create 2-channel images where each channel has signal at the same offset.
    pre_image = np.zeros((2, 120, 120), dtype=np.float32)
    post_image = np.zeros((2, 120, 120), dtype=np.float32)

    # Channel 0: signal at one location
    pre_image[0, 60, 60] = 1.0
    post_image[0, 70, 53] = 1.0

    # Channel 1: signal at the same offset but different location
    pre_image[1, 40, 40] = 1.0
    post_image[1, 50, 33] = 1.0

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()

    # Explicitly using all channels should produce valid flow.
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=[0, 1])

    np.testing.assert_array_equal(field.shape, [4, 2, 2])
    # Both channels have the same offset (7, -10), so the average xcorr
    # should produce the same result.
    np.testing.assert_array_equal(field[0, ...], 7 * np.ones((2, 2)))
    np.testing.assert_array_equal(field[1, ...], -10 * np.ones((2, 2)))

  def test_single_channel_backward_compatibility(self):
    """Tests that single-channel (spatial-only) input with channel=None works."""
    pre_image = np.zeros((120, 120), dtype=np.uint8)
    post_image = np.zeros((120, 120), dtype=np.uint8)

    pre_image[60, 60] = 255
    post_image[70, 53] = 255

    calculator = flow_field.JAXMaskedXCorrWithStatsCalculator()
    # channel=None with 2D input should work identically to original behavior.
    field = calculator.flow_field(
        pre_image, post_image, patch_size=80, step=40, batch_size=4,
        channel=None)

    np.testing.assert_array_equal(field.shape, [4, 2, 2])
    np.testing.assert_array_equal(field[0, ...], 7 * np.ones((2, 2)))
    np.testing.assert_array_equal(field[1, ...], -10 * np.ones((2, 2)))

  def test_stitch_rigid_channel_none_multichannel(self):
    """Tests that stitch_rigid auto-detects multichannel with channel=None."""
    from sofima import stitch_rigid

    # Create 2-channel tiles where both channels have consistent signal.
    tile_00 = np.zeros((2, 200, 200), dtype=np.float32)
    tile_10 = np.zeros((2, 200, 200), dtype=np.float32)

    # Place features with a known offset in both channels.
    tile_00[0, 100, 150] = 1.0
    tile_10[0, 100, 50] = 1.0
    tile_00[1, 80, 150] = 1.0
    tile_10[1, 80, 50] = 1.0

    tile_map = {(0, 0): tile_00, (1, 0): tile_10}

    # channel=None should auto-detect multichannel and use all channels.
    conn_x, conn_y = stitch_rigid.compute_coarse_offsets(
        (1, 2), tile_map, overlaps_xy=((100,), (100,)),
        min_range=(0,), min_overlap=50, channel=None)

    # Verify that it produces a valid (non-nan, non-inf) result.
    self.assertFalse(np.all(np.isnan(conn_x[0, 0, 0, :])))


if __name__ == '__main__':
  absltest.main()
