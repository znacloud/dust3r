#
# Copyright (C) 2024, Nianan Zeng
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  znacloud@gmail.com
#

from dust3r.utils.device import to_numpy
from pathlib import Path
import numpy as np
import PIL


def export_optimized_scene(outdir, imgs, pts3d, mask, focals, cams2world):
    """
    Export the point clouds along with its optimized intrinsic and extrinsic datasets.

    The Output file tree is as bellow:

    -- DUSt3R_Scene
        -- ref_imgs
            -- img_01.jpg
            -- img_02.jpg
            -- img
        -- camera_intrinsics.json
        -- camera_extrinsics.json
        -- conf_points3D.txt

    """
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene_dir = Path(outdir) / "DUSt3R_Scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Export images to files
    _export_imgs(scene_dir, imgs)


def _export_imgs(scene_dir: Path, imgs):
    ref_imgs_dir = scene_dir / "ref_imgs"
    ref_imgs_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(imgs)):
        image = imgs[i]
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)

        image = PIL.Image.fromarray(image)
        image.save(ref_imgs_dir / f"img_{i:03}.jpg")
