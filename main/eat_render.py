"""Render EATalker prediction npz to video, based on render.py."""

import cv2
import numpy as np
import os
import subprocess
import sys
import torch
from tqdm import tqdm

from EATalker.dataset.dataset_config import dataset_config


def render_meshes(mesh_vertices: torch.Tensor,
                  faces: torch.Tensor,
                  img_size: tuple = (256, 256),
                  aa_factor: int = 1,
                  vis_bs: int = 100):
    # 渲染阶段：把预测的顶点序列喂给 PyTorch3D，生成一帧帧 RGB 图像
    assert len(mesh_vertices) == len(faces) or len(faces) == 1
    if len(faces) == 1:
        faces = faces.repeat(len(mesh_vertices), 1, 1)

    x_max = y_max = mesh_vertices[..., 0:2].abs().max()
    from pytorch3d.renderer import (
        DirectionalLights,
        FoVOrthographicCameras,
        HardPhongShader,
        Materials,
        MeshRasterizer,
        MeshRenderer,
        RasterizationSettings,
        TexturesVertex,
    )
    from pytorch3d.renderer.blending import BlendParams
    from pytorch3d.structures import Meshes
    aa_factor = int(aa_factor)
    device = mesh_vertices.device
    verts_batch_lst = torch.split(mesh_vertices, vis_bs)
    faces_batch_lst = torch.split(faces, vis_bs)
    R = torch.eye(3).to(mesh_vertices.device)
    R[0, 0] = -1
    R[2, 2] = -1
    T = torch.zeros(3).to(mesh_vertices.device)
    T[2] = 10
    R, T = R[None], T[None]
    W, H = img_size
    cameras = FoVOrthographicCameras(
        device=device,
        R=R,
        T=T,
        znear=0.01,
        zfar=3,
        max_x=x_max * 1.2,
        min_x=-x_max * 1.2,
        max_y=y_max * 1.2,
        min_y=-y_max * 1.2,
    )

    raster_settings = RasterizationSettings(
        image_size=(W * aa_factor, H * aa_factor),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    raster = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings,
    )
    blend_params = BlendParams(
        sigma=0.0, gamma=0.0, background_color=(0.0, 0.0, 0.0))
    lights = DirectionalLights(
        device=device,
        direction=((0, 0, 1), ),
        ambient_color=((0.3, 0.3, 0.3), ),
        diffuse_color=((0.6, 0.6, 0.6), ),
        specular_color=((0.1, 0.1, 0.1), ))
    materias = Materials(
        ambient_color=((1, 1, 1), ),
        diffuse_color=((1, 1, 1), ),
        specular_color=((1, 1, 1), ),
        shininess=15,
        device=device)
    shader = HardPhongShader(
        device=device,
        cameras=cameras,
        lights=lights,
        materials=materias,
        blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=raster,
        shader=shader,
    )
    rendered_imgs = []
    for verts_batch, faces_batch in tqdm(
            zip(verts_batch_lst, faces_batch_lst),
            total=(len(verts_batch_lst))):
        textures = TexturesVertex(verts_features=torch.ones_like(verts_batch))
        meshes = Meshes(
            verts=verts_batch, faces=faces_batch, textures=textures)
        with torch.no_grad():
            imgs = renderer(meshes)
        rendered_imgs.append(imgs.cpu())
    rendered_imgs = torch.cat(rendered_imgs, dim=0)
    if aa_factor > 1:
        rendered_imgs = rendered_imgs.permute(0, 3, 1, 2)
        rendered_imgs = torch.nn.functional.interpolate(
            rendered_imgs, scale_factor=1 / aa_factor, mode='bicubic')
        rendered_imgs = rendered_imgs.permute(0, 2, 3, 1)
    return rendered_imgs


def read_obj(in_path):
    # 从 OBJ 文件中解析出顶点和三角面索引，用于后续渲染
    with open(in_path, 'r') as obj_file:
        lines = obj_file.readlines()

    verts = []
    faces = []
    for line in lines:
        line = line.strip()
        elements = line.split()

        if len(elements) == 0:
            continue

        if elements[0] == 'v':
            x, y, z = map(float, elements[1:4])
            verts.append((x, y, z))
        elif elements[0] == 'f':
            face_indices = [int(index.split('/')[0]) for index in elements[1:]]
            faces.append(face_indices)
    return np.array(verts), np.array(faces)


def pad_for_libx264(image_array):
    if image_array.ndim == 2 or \
      (image_array.ndim == 3 and image_array.shape[2] == 3):
        hei_index = 0
        wid_index = 1
    elif image_array.ndim == 4 or \
      (image_array.ndim == 3 and image_array.shape[2] != 3):
        hei_index = 1
        wid_index = 2
    else:
        return image_array
    hei_pad = image_array.shape[hei_index] % 2
    wid_pad = image_array.shape[wid_index] % 2
    if hei_pad + wid_pad > 0:
        pad_width = []
        for dim_index in range(image_array.ndim):
            if dim_index == hei_index:
                pad_width.append((0, hei_pad))
            elif dim_index == wid_index:
                pad_width.append((0, wid_pad))
            else:
                pad_width.append((0, 0))
        values = 0
        image_array = \
         np.pad(image_array,
             pad_width,
             mode='constant', constant_values=values)
    return image_array


def array_to_video(
    image_array: np.ndarray,
    output_path: str,
    fps=30,
    resolution=None,
    disable_log: bool = False,
) -> None:
    # 利用管道把逐帧图像送入 ffmpeg，压成带视频编码的 mp4
    if not isinstance(image_array, np.ndarray):
        raise TypeError('Input should be np.ndarray.')
    assert image_array.ndim == 4
    assert image_array.shape[-1] == 3
    if resolution:
        height, width = resolution
        width += width % 2
        height += height % 2
    else:
        image_array = pad_for_libx264(image_array)
        height, width = image_array.shape[1], image_array.shape[2]
    ffmpeg_exec = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
    command = [
        ffmpeg_exec,
        '-y',
        '-f',
        'rawvideo',
        '-s',
        f'{int(width)}x{int(height)}',
        '-pix_fmt',
        'bgr24',
        '-r',
        f'{fps}',
        '-loglevel',
        'error',
        '-threads',
        '4',
        '-i',
        '-',
        '-vcodec',
        'libx264',
        '-an',
        output_path,
    ]
    if not disable_log:
        print(f'Running "{" ".join(command)}"')
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stderr is None:
        raise BrokenPipeError('No buffer received.')
    index = 0
    while True:
        if index >= image_array.shape[0]:
            break
        process.stdin.write(image_array[index].tobytes())
        index += 1
    process.stdin.close()
    process.stderr.close()
    process.wait()


def get_obj_faces(annot_type: str):
    # Only `inhouse_blendshape_weight` is needed for the current EATalker dataset.
    # Uncomment the entries below if you plan to render predictions for other datasets.
    template_obj_path_dict = {
        # '3DETF_blendshape_weight': 'resources/obj_template/3DETF_blendshape_weight.obj',
        # 'FLAME_5023_vertices': 'resources/obj_template/FLAME_5023_vertices.obj',
        # 'BIWI_23370_vertices': 'resources/obj_template/BIWI_23370_vertices.obj',
        # 'flame_params_from_dadhead': 'resources/obj_template/flame_params_from_dadhead.obj',
        'inhouse_blendshape_weight': 'resources/obj_template/inhouse_blendshape_weight.obj',
        # 'meshtalk_6172_vertices': 'resources/obj_template/meshtalk_6172_vertices.obj'
    }

    faces = np.array(read_obj(template_obj_path_dict[annot_type])[1])
    if faces.min() == 1:
        faces = faces - 1
    return faces


def main():
    if len(sys.argv) < 4:
        print('Usage: python eat_render.py <pred_npz> <audio_dir> <output_dir>')
        return
    npz_path, audio_dir, out_dir = sys.argv[1:4]
    # 载入预测 npz：每个音频键下存着不同数据集的顶点序列
    os.makedirs(out_dir, exist_ok=True)
    out_dict = np.load(npz_path, allow_pickle=True)
    print(list(out_dict.keys()))
    for wav_f in out_dict.keys():
        wav_path = os.path.join(audio_dir, wav_f)
        cur_out_dict = out_dict[wav_f]
        if isinstance(cur_out_dict, np.ndarray):
            cur_out_dict = cur_out_dict.item()
        for datasetname, out in cur_out_dict.items():
            out_tensor = torch.tensor(out)
            annot_type = dataset_config[datasetname]['annot_type']
            faces = get_obj_faces(annot_type)
            with torch.no_grad():
                # 将预测顶点渲染为图像帧
                out_size = (512, 512)
                img_array = render_meshes(out_tensor.cuda(),
                                          torch.tensor(faces[None],
                                                       dtype=torch.long).cuda(),
                                          out_size)
                img_array = (img_array[..., :3].cpu().numpy() * 255).astype(
                    np.uint8)
            out_key = f'{wav_f[:-4]}'
            out_video_path = os.path.join(out_dir,
                                          f'{out_key}_{datasetname}.mp4')
            out_video_dir = os.path.dirname(out_video_path)
            os.makedirs(out_video_dir, exist_ok=True)
            tmp_path = os.path.join(out_video_dir, 'tmp.mp4')
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            fps = 25 if annot_type == 'BIWI_23370_vertices' else 30
            array_to_video(img_array, output_path=tmp_path, fps=fps)

            # 第二次调用 ffmpeg，把视频与原始音频合成
            cmd = f'ffmpeg -i {tmp_path} -i  {wav_path} -c:v copy -c:a aac -shortest -strict -2  {out_video_path} -y '
            subprocess.run(cmd, shell=True)
            os.remove(tmp_path)


if __name__ == '__main__':
    main()
