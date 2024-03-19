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
from dust3r.viz import OPENGL
from pathlib import Path
import numpy as np
import PIL
import json


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.uint8)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def export_optimized_scene(outdir, imgs, pts3d, confs, masks, focals, cams2world):
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
    # pts3d = to_numpy(pts3d) # already be numpy format
    # confs = to_numpy(confs) # already be numpy format
    # masks = to_numpy(masks) # already be numpy format
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene_dir = Path(outdir) / "DUSt3R_Scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    # >>> pts3d.shape=(4, 512, 288, 3), imgs.shape=(4, 512, 288, 3)
    # >>> focals.shape=(4, 1), pose.shape=(4, 4, 4)
    # print(f">>> confs.shape={np.shape(confs)}")
    # >>> confs.shape=(4, 512, 288)

    # Export images & intrinsics & extrinsic to files
    _export_camera_infos(scene_dir, imgs, focals, poses=cams2world)

    # Export PointsCloud
    _export_pointscloud(scene_dir, pts3d, imgs, confs, masks)


def _export_camera_infos(scene_dir: Path, imgs, focals, poses):

    assert len(focals) == len(imgs) == len(poses), "data dimention mismatch!"

    ref_imgs_dir = scene_dir / "ref_imgs"
    ref_imgs_dir.mkdir(parents=True, exist_ok=True)

    intr_file = scene_dir / "camera_intrinsics.json"
    intr_obj = {}

    extr_file = scene_dir / "camera_extrinsics.json"
    extr_obj = {}

    for i in range(len(imgs)):
        image = imgs[i]
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)

        idx = f"{i:03}"
        image_name = f"img_{idx}.jpg"

        # Save images
        image = PIL.Image.fromarray(image)
        image.save(ref_imgs_dir / image_name)

        # Save intrinsic
        focal = focals[i]
        if focal is None:
            focal = min(H, W) * 1.1  # default value
        elif isinstance(focal, np.ndarray):
            focal = focal[0]

        intr_obj[idx] = dict(
            id=idx,
            model="PINHOLE",
            width=W,
            height=H,
            focal_x=focal,
            focal_y=focal,
            pp_x=0,
            pp_y=0,
        )
        with open(intr_file, "w") as f:
            json.dump(intr_obj, f, cls=NumpyEncoder, indent=2)

        # Save extrinsic
        pose = poses[i] @ OPENGL  # Transformed to OPENGL format
        R = pose[:3, :3]
        T = pose[:3, 3:4]

        # Test
        # print(f"{R=},{T=}")
        # print(f"rflat={R.flatten()},tvect={T.flatten()}")
        # Test end

        extr_obj[idx] = dict(
            id=idx,
            camera_id=idx,
            img_name=image_name,
            rflat=R.flatten(),
            tvec=T.flatten(),
        )
        with open(extr_file, "w") as f:
            json.dump(extr_obj, f, cls=NumpyEncoder, indent=2)


def _export_pointscloud(scene_dir, pts3d, imgs, confs, masks):
    assert len(pts3d) == len(imgs) == len(confs), "data dimention mismatch!"

    pts3d_file = scene_dir / "conf_points3D.txt"

    with open(pts3d_file, "w") as f:
        for i in range(len(imgs)):
            image_name = f"img_{i:03}.jpg"
            f.writelines([f"# ref_img: {image_name}\n"])

            img_pts = pts3d[i]  # HxWx3
            img_col = imgs[i]  # HxWx3
            if img_col.dtype != np.uint8:
                img_col = np.uint8(255 * img_col)

            conf = confs[i]  # HxW
            msk = masks[i]  # HxW
            pts_cols = np.concatenate(
                (
                    img_pts.reshape(-1, 3),
                    img_col.reshape(-1, 3).astype(int),
                    conf.reshape(-1, 1),
                    msk.reshape(-1, 1).astype(int),
                ),
                axis=-1,
            )
            # Join the strings within each row
            string_array = [",".join(map(str, row)) + "\n" for row in pts_cols]
            f.writelines(string_array)
