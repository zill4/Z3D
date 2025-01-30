import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import numpy as np
import json
import trimesh

from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from .hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2DConditionModel, UNet2p5DConditionModel
from .hy3dgen.texgen.hunyuanpaint.pipeline import HunyuanPaintPipeline

from diffusers import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler, 
    PNDMScheduler, 
    DPMSolverMultistepScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    LCMScheduler
    )

scheduler_mapping = {
    "DPM++": DPMSolverMultistepScheduler,
    "DPM++SDE": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "LCMScheduler": LCMScheduler
}
available_schedulers = list(scheduler_mapping.keys())
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

from .utils import log, print_memory

class ComfyProgressCallback:
    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        
    def __call__(self, pipe, i, t, callback_kwargs):
        self.pbar.update(1)
        return {
            "latents": callback_kwargs["latents"],
            "prompt_embeds": callback_kwargs["prompt_embeds"],
            "negative_prompt_embeds": callback_kwargs["negative_prompt_embeds"]
        }

class Hy3DTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_vae": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
            },
        }
    RETURN_TYPES = ("HY3DCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer, compile_vae):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_transformer": compile_transformer,
            "compile_vae": compile_vae,
        }

        return (compile_args, )
    
#region Model loading
class Hy3DModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            },
            "optional": {
                "compile_args": ("HY3DCOMPILEARGS", {"tooltip": "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("HY3DMODEL", "HY3DVAE")
    RETURN_NAMES = ("pipeline", "vae")
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model, compile_args=None, attention_mode="sdpa"):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        config_path = os.path.join(script_directory, "configs", "dit_config.yaml")
        model_path = folder_paths.get_full_path("diffusion_models", model)
        pipe, vae = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=model_path, 
            config_path=config_path, 
            use_safetensors=True, 
            device=device, 
            offload_device=offload_device,
            compile_args=compile_args,
            attention_mode=attention_mode)
        
        return (pipe, vae,)

class DownloadAndLoadHy3DDelightModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-delight-v2-0"],),
            },
        }

    RETURN_TYPES = ("HY3DDIFFUSERSPIPE",)
    RETURN_NAMES = ("delight_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=["*hunyuan3d-delight-v2-0*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

        delight_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        delight_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(delight_pipe.scheduler.config)
        delight_pipe = delight_pipe.to(device, torch.float16)
        delight_pipe.enable_model_cpu_offload()
        
        return (delight_pipe,)
        
class Hy3DDelightImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delight_pipe": ("HY3DDIFFUSERSPIPE",),
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 50, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "cfg_image": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "cfg_text": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "scheduler": ("NOISESCHEDULER",),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, delight_pipe, image, width, height, cfg_image, cfg_text, steps, seed, scheduler=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        if scheduler is not None:
            if not hasattr(self, "default_scheduler"):
                self.default_scheduler = delight_pipe.scheduler
            delight_pipe.scheduler = scheduler
        else:
            if hasattr(self, "default_scheduler"):
                delight_pipe.scheduler = self.default_scheduler

        image = image.permute(0, 3, 1, 2).to(device)

        image = delight_pipe(
            prompt="",
            image=image,
            generator=torch.manual_seed(seed),
            height=height,
            width=width,
            num_inference_steps=steps,
            image_guidance_scale=cfg_image,
            guidance_scale=cfg_text,
            output_type="pt",
            
        ).images[0]

        out_tensor = image.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensor, )
    
class DownloadAndLoadHy3DPaintModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-paint-v2-0"],),
            },
        }

    RETURN_TYPES = ("HY3DDIFFUSERSPIPE",)
    RETURN_NAMES = ("multiview_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=[f"*{model}*"],
                ignore_patterns=["*diffusion_pytorch_model.bin"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        torch_dtype = torch.float16
        config_path = os.path.join(model_path, 'unet', 'config.json')
        unet_ckpt_path_safetensors = os.path.join(model_path, 'unet','diffusion_pytorch_model.safetensors')
        unet_ckpt_path_bin = os.path.join(model_path, 'unet','diffusion_pytorch_model.bin')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        

        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        with init_empty_weights():
            unet = UNet2DConditionModel(**config)
            unet = UNet2p5DConditionModel(unet)

        # Try loading safetensors first, fall back to .bin
        if os.path.exists(unet_ckpt_path_safetensors):
            import safetensors.torch
            unet_sd = safetensors.torch.load_file(unet_ckpt_path_safetensors)
        elif os.path.exists(unet_ckpt_path_bin):
            unet_sd = torch.load(unet_ckpt_path_bin, map_location='cpu', weights_only=True)
        else:
            raise FileNotFoundError(f"No checkpoint found at {unet_ckpt_path_safetensors} or {unet_ckpt_path_bin}")

        #unet.load_state_dict(unet_ckpt, strict=True)
        for name, param in unet.named_parameters():
            set_module_tensor_to_device(unet, name, device=offload_device, dtype=torch_dtype, value=unet_sd[name])

        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", device=device, torch_dtype=torch_dtype)
        clip = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")

        pipeline = HunyuanPaintPipeline(
            unet=unet,
            vae = vae,
            text_encoder=clip,
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            )

        pipeline.enable_model_cpu_offload()
        return (pipeline,)

#region Texture
class Hy3DCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
                "camera_distance": ("FLOAT", {"default": 1.45, "min": 0.1, "max": 10.0, "step": 0.001}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("HY3DCAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights, camera_distance, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "camera_distance": camera_distance,
            "ortho_scale": ortho_scale,
            }
        
        return (camera_config,)
    
class Hy3DMeshUVWrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
            },
        }

    RETURN_TYPES = ("HY3DMESH", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh):
        from .hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
        mesh = mesh_uv_wrap(mesh)
        
        return (mesh,)

class Hy3DRenderMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "texture_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "normal_space": (["world", "tangent"], {"default": "world"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MESHRENDER")
    RETURN_NAMES = ("normal_maps", "position_maps", "renderer")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, render_size, texture_size, camera_config=None, normal_space="world"):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            camera_distance = 1.45
            ortho_scale = 1.2
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            camera_distance = camera_config["camera_distance"]
            ortho_scale = camera_config["ortho_scale"]
        
        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size,
            camera_distance=camera_distance,
            ortho_scale=ortho_scale)

        self.render.load_mesh(mesh)

        if normal_space == "world":
            normal_maps = self.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
            normal_tensors = torch.stack(normal_maps, dim=0)
        elif normal_space == "tangent":
            normal_maps = self.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, bg_color=[0, 0, 0], use_abs_coor=False)
            normal_tensors = torch.stack(normal_maps, dim=0)
            normal_tensors = 2.0 * normal_tensors - 1.0  # Map [0,1] to [-1,1]
            normal_tensors = normal_tensors / (torch.norm(normal_tensors, dim=-1, keepdim=True) + 1e-6)
            # Remap axes for standard normal map convention
            image = torch.zeros_like(normal_tensors)
            image[..., 0] = normal_tensors[..., 0]  # View right to R
            image[..., 1] = normal_tensors[..., 1]  # View up to G
            image[..., 2] = -normal_tensors[..., 2] # View forward (negated) to B
            normal_tensors = (image + 1) * 0.5
        
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        position_tensors = torch.stack(position_maps, dim=0)
        
        return (normal_tensors, position_tensors, self.render,)
    
    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True, bg_color=[1, 1, 1]):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map, _ = self.render.render_normal(
                elev, azim, bg_color=bg_color, use_abs_coor=use_abs_coor, return_type='th')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='th')
            position_maps.append(position_map)

        return position_maps
    
class Hy3DRenderSingleView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "render_type": (["normal", "depth"], {"default": "normal"}),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distance": ("FLOAT", {"default": 1.45, "min": 0.1, "max": 10.0, "step": 0.001}),
                "pan_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "pan_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.001}),
                "azimuth": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "elevation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "bg_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, render_type, camera_type, ortho_scale, camera_distance, pan_x, pan_y, render_size, azimuth, elevation, bg_color):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        bg_color = [int(x.strip())/255.0 for x in bg_color.split(",")]

        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=1024,
            camera_distance=camera_distance,
            camera_type=camera_type,
            ortho_scale=ortho_scale,
            filter_mode='linear'
            )

        self.render.load_mesh(mesh)

        if render_type == "normal":
            normals, mask = self.render.render_normal(
                elevation,
                azimuth,
                camera_distance=camera_distance,
                center=None,
                resolution=render_size,
                bg_color=[0, 0, 0],
                use_abs_coor=False,
                pan_x=pan_x,
                pan_y=pan_y
            )

            normals = 2.0 * normals - 1.0  # Map [0,1] to [-1,1]
            normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
            # Remap axes for standard normal map convention
            image = torch.zeros_like(normals)
            image[..., 0] = normals[..., 0]  # View right to R
            image[..., 1] = normals[..., 1]  # View up to G
            image[..., 2] = -normals[..., 2] # View forward (negated) to B

            image = (image + 1) * 0.5

            #mask = mask.cpu().float()
            masked_image = image * mask

            bg_color = torch.tensor(bg_color, dtype=torch.float32, device=image.device)
            bg = bg_color.view(1, 1, 3) * (1.0 - mask)
            final_image = masked_image + bg
        elif render_type == "depth":
            depth = self.render.render_depth(
                elevation,
                azimuth,
                camera_distance=camera_distance,
                center=None,
                resolution=render_size,
                pan_x=pan_x,
                pan_y=pan_y
            )
            final_image = depth.unsqueeze(0).repeat(1, 1, 1, 3).cpu().float()
        
        return (final_image,)
    
    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map, _ = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='th')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='th')
            position_maps.append(position_map)

        return position_maps
    
class Hy3DRenderMultiViewDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "texture_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("depth_maps", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, render_size, texture_size, camera_config=None):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            camera_distance = 1.45
            ortho_scale = 1.2
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            camera_distance = camera_config["camera_distance"]
            ortho_scale = camera_config["ortho_scale"]

        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size,
            camera_distance=camera_distance,
            ortho_scale=ortho_scale)

        self.render.load_mesh(mesh)

       

        depth_maps = self.render_depth_multiview(
            selected_camera_elevs, selected_camera_azims)
        depth_tensors = torch.stack(depth_maps, dim=0)
        depth_tensors = depth_tensors.repeat(1, 1, 1, 3)
        
        return (depth_tensors,)
    
    def render_depth_multiview(self, camera_elevs, camera_azims):
        depth_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):        
            depth_map = self.render.render_depth(elev, azim, return_type='th')
            depth_maps.append(depth_map)

        return depth_maps

class Hy3DDiffusersSchedulerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DDIFFUSERSPIPE",),
                "scheduler": (available_schedulers,
                    {
                        "default": 'Euler A'
                    }),
                "sigmas": (["default", "karras", "exponential", "beta"],),
            },
        }

    RETURN_TYPES = ("NOISESCHEDULER",)
    RETURN_NAMES = ("diffusers_scheduler",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, scheduler, sigmas):

        scheduler_config = dict(pipeline.scheduler.config)
        
        if scheduler in scheduler_mapping:
            if scheduler == "DPM++SDE":
                scheduler_config["algorithm_type"] = "sde-dpmsolver++"
            else:
                scheduler_config.pop("algorithm_type", None)
            if sigmas == "default":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "karras":
                scheduler_config["use_karras_sigmas"] = True
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "exponential":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = True
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "beta":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = True
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        return (noise_scheduler,)
    
class Hy3DSampleMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DDIFFUSERSPIPE",),
                "ref_image": ("IMAGE", ),
                "normal_maps": ("IMAGE", ),
                "position_maps": ("IMAGE", ),
                "view_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "scheduler": ("NOISESCHEDULER",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, ref_image, normal_maps, position_maps, view_size, seed, steps, camera_config=None, scheduler=None):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        torch.manual_seed(seed)
        generator=torch.Generator(device=pipeline.device).manual_seed(seed)

        input_image = ref_image.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        device = mm.get_torch_device()

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
        
        camera_info = [(((azim // 30) + 9) % 12) // {-90: 3, -45: 2, -20: 1, 0: 1, 20: 1, 45: 2, 90: 3}[
            elev] + {-90: 36, -45: 30, -20: 0, 0: 12, 20: 24, 45: 30, 90: 40}[elev] for azim, elev in
                    zip(selected_camera_azims, selected_camera_elevs)]
        print(camera_info)
        
        normal_maps_np = (normal_maps * 255).to(torch.uint8).cpu().numpy()
        normal_maps_pil = [Image.fromarray(normal_map) for normal_map in normal_maps_np]

        position_maps_np = (position_maps * 255).to(torch.uint8).cpu().numpy()
        position_maps_pil = [Image.fromarray(position_map) for position_map in position_maps_np]
        
        control_images = normal_maps_pil + position_maps_pil

        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((view_size, view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        callback = ComfyProgressCallback(total_steps=steps)

        if scheduler is not None:
            if not hasattr(self, "default_scheduler"):
                self.default_scheduler = pipeline.scheduler
            pipeline.scheduler = scheduler
        else:
            if hasattr(self, "default_scheduler"):
                pipeline.scheduler = self.default_scheduler

        multiview_images = pipeline(
            input_image,
            width=view_size,
            height=view_size,
            generator=generator,
            num_in_batch = num_view,
            camera_info_gen = [camera_info],
            camera_info_ref = [[0]],
            normal_imgs = normal_image,
            position_imgs = position_image,
            num_inference_steps=steps,
            output_type="pt",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"]
            ).images

        out_tensors = multiview_images.permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensors,)
    
class Hy3DBakeFromMultiview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER") 
    RETURN_NAMES = ("texture", "mask", "renderer")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, images, renderer, camera_config=None):
        device = mm.get_torch_device()
        self.render = renderer

        multiviews = images.permute(0, 3, 1, 2)
        multiviews = multiviews.cpu().numpy()
        multiviews_pil = [Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8)) for image in multiviews]

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            selected_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            selected_view_weights = camera_config["selected_view_weights"]

        merge_method = 'fast'
        self.bake_exp = 4
        
        texture, mask = self.bake_from_multiview(multiviews_pil,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=merge_method)
        
        
        mask = mask.squeeze(-1).cpu().float()
        texture = texture.unsqueeze(0).cpu().float()

        return (texture, mask, self.render)
    
    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        pbar = ProgressBar(len(views))
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            pbar.update(1)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8
    
class Hy3DMeshVerticeInpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER" ) 
    RETURN_NAMES = ("texture", "mask", "renderer" )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, texture, renderer, mask):
        from .hy3dgen.texgen.differentiable_renderer.mesh_processor import meshVerticeInpaint
        vtx_pos, pos_idx, vtx_uv, uv_idx = renderer.get_mesh()

        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = texture.squeeze(0).cpu().numpy() * 255

        texture_np, mask_np = meshVerticeInpaint(
            texture_np, mask_np, vtx_pos, vtx_uv, pos_idx, uv_idx)
            
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return (texture_tensor, mask_tensor, renderer)

class CV2InpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "inpaint_radius": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "inpaint_method": (["ns", "telea"], {"default": "ns"}),
            },
        }

    RETURN_TYPES = ("IMAGE", ) 
    RETURN_NAMES = ("texture", )
    FUNCTION = "inpaint"
    CATEGORY = "Hunyuan3DWrapper"

    def inpaint(self, texture, mask, inpaint_radius, inpaint_method):
        import cv2
        mask = 1 - mask
        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = (texture.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        if inpaint_method == "ns":
            inpaint_algo = cv2.INPAINT_NS
        elif inpaint_method == "telea":
            inpaint_algo = cv2.INPAINT_TELEA
            
        texture_np = cv2.inpaint(
            texture_np,
            mask_np,
            inpaint_radius,
            inpaint_algo)
        
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)
        
        return (texture_tensor, )
    
class Hy3DApplyTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("HY3DMESH", ) 
    RETURN_NAMES = ("mesh", )
    FUNCTION = "apply"
    CATEGORY = "Hunyuan3DWrapper"

    def apply(self, texture, renderer):
        self.render = renderer
        self.render.set_texture(texture.squeeze(0))
        textured_mesh = self.render.save_mesh()
        
        return (textured_mesh,)

#region Mesh

class Hy3DLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):
        
        mesh = trimesh.load(glb_path, force="mesh")
        
        return (mesh,)


class Hy3DGenerateMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DMODEL",),
                "image": ("IMAGE", ),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, image, steps, guidance_scale, seed, mask=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)
        image = image * 2 - 1

        if mask is not None:
            mask = mask.unsqueeze(0).to(device)

        pipeline.to(device)

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        latents = pipeline(
            image=image, 
            mask=mask,
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed))

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        pipeline.to(offload_device)
        
        return (latents, )
    
class Hy3DVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HY3DVAE",),
                "latents": ("HY3DLATENT", ),
                "box_v": ("FLOAT", {"default": 1.01, "min": -10.0, "max": 10.0, "step": 0.001}),
                "octree_resolution": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 16}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 10000000, "step": 1}),
                "mc_level": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
            },
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, vae, latents, box_v, octree_resolution, mc_level, num_chunks, mc_algo):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae.to(device)
        latents = 1. / vae.scale_factor * latents
        latents = vae(latents)
        
        outputs = vae.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
        )[0]
        vae.to(offload_device)

        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
        log.info(f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces")
        
        return (mesh_output, )
    
class Hy3DPostprocessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
                "smooth_normals": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, remove_floaters, remove_degenerate_faces, reduce_faces, max_facenum, smooth_normals):
        new_mesh = mesh.copy()
        if remove_floaters:
            new_mesh = FloaterRemover()(new_mesh)
            log.info(f"Removed floaters, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if remove_degenerate_faces:
            new_mesh = DegenerateFaceRemover()(new_mesh)
            log.info(f"Removed degenerate faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if reduce_faces:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)
            log.info(f"Reduced faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if smooth_normals:              
            new_mesh.vertex_normals = trimesh.smoothing.get_vertices_normals(new_mesh)

        
        return (new_mesh, )

class Hy3DFastSimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "target_count": ("INT", {"default": 40000, "min": 1, "max": 100000000, "step": 1, "tooltip": "Target number of triangles"}),
                "aggressiveness": ("INT", {"default": 7, "min": 0, "max": 100, "step": 1, "tooltip": "Parameter controlling the growth rate of the threshold at each iteration when lossless is False."}),
                "max_iterations": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1, "tooltip": "Maximal number of iterations"}),
                "update_rate": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of iterations between each update"}),
                "preserve_border": ("BOOLEAN", {"default": True, "tooltip": "Flag for preserving the vertices situated on open borders."}),
                "lossless": ("BOOLEAN", {"default": False, "tooltip": "Flag for using the lossless simplification method. Sets the update rate to 1"}),
                "threshold_lossless": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Threshold for the lossless simplification method."}),
            },
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Simplifies the mesh using Fast Quadric Mesh Reduction: https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction"

    def process(self, mesh, target_count, aggressiveness, preserve_border, max_iterations,lossless, threshold_lossless, update_rate):
        new_mesh = mesh.copy()
        try:
            import pyfqmr
        except ImportError:
            raise ImportError("pyfqmr not found. Please install it using 'pip install pyfqmr' https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction")
        
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
        mesh_simplifier.simplify_mesh(
            target_count=target_count, 
            aggressiveness=aggressiveness,
            update_rate=update_rate,
            max_iterations=max_iterations,
            preserve_border=preserve_border, 
            verbose=True,
            lossless=lossless
            )
        new_mesh.vertices, new_mesh.faces, _ = mesh_simplifier.getMesh()
        log.info(f"Simplified mesh to {target_count} vertices, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")   
        
        return (new_mesh, )
    
class Hy3DMeshInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
            },
        }

    RETURN_TYPES = ("HY3DMESH", "INT", "INT", )
    RETURN_NAMES = ("mesh", "vertices", "faces",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh):
        vertices_count = mesh.vertices.shape[0]
        faces_count = mesh.faces.shape[0]
        log.info(f"Hy3DMeshInfo: Mesh has {vertices_count} vertices and {mesh.faces.shape[0]} faces")
        return {"ui": {
            "text": [f"{vertices_count:,.0f}x{faces_count:,.0f}"]}, 
            "result": (mesh, vertices_count, faces_count) 
        }
    
class Hy3DIMRemesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "merge_vertices": ("BOOLEAN", {"default": True}),
                "vertex_count": ("INT", {"default": 10000, "min": 100, "max": 10000000, "step": 1}),
                "smooth_iter": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "align_to_boundaries": ("BOOLEAN", {"default": True}),
                "triangulate_result": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "remesh"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Remeshes the mesh using instant-meshes: https://github.com/wjakob/instant-meshes, Note: this will remove all vertex colors and textures."

    def remesh(self, mesh, merge_vertices, vertex_count, smooth_iter, align_to_boundaries, triangulate_result):
        try:
            import pynanoinstantmeshes as PyNIM
        except ImportError:
            raise ImportError("pynanoinstantmeshes not found. Please install it using 'pip install pynanoinstantmeshes'")
        new_mesh = mesh.copy()
        if merge_vertices:
            mesh.merge_vertices(new_mesh)

        new_verts, new_faces = PyNIM.remesh(
            np.array(mesh.vertices, dtype=np.float32),
            np.array(mesh.faces, dtype=np.uint32),
            vertex_count,
            align_to_boundaries=align_to_boundaries,
            smooth_iter=smooth_iter
        )
        if new_verts.shape[0] - 1 != new_faces.max():
            # Skip test as the meshing failed
            raise ValueError("Instant-meshes failed to remesh the mesh")
        new_verts = new_verts.astype(np.float32)
        if triangulate_result:
            new_faces = trimesh.geometry.triangulate_quads(new_faces)

        new_mesh = trimesh.Trimesh(new_verts, new_faces)
        
        return (new_mesh, )
    
class Hy3DGetMeshPBRTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "texture" : (["base_color", "emissive", "metallic_roughness", "normal", "occlusion"], ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "get_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def get_textures(self, mesh, texture):
        
        TEXTURE_MAPPING = {
            'base_color': ('baseColorTexture', "Base color"),
            'emissive': ('emissiveTexture', "Emissive"),
            'metallic_roughness': ('metallicRoughnessTexture', "Metallic roughness"),
            'normal': ('normalTexture', "Normal"),
            'occlusion': ('occlusionTexture', "Occlusion"),
        }
        
        texture_attr, texture_name = TEXTURE_MAPPING[texture]
        texture_data = getattr(mesh.visual.material, texture_attr)
        
        if texture_data is None:
            raise ValueError(f"{texture_name} texture not found")
            
        to_tensor = transforms.ToTensor()
        return (to_tensor(texture_data).unsqueeze(0).permute(0, 2, 3, 1).cpu().float(),)
    
class Hy3DSetMeshPBRTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "image": ("IMAGE", ),
                "texture" : (["base_color", "emissive", "metallic_roughness", "normal", "occlusion"], ),
            },
        }

    RETURN_TYPES = ("HY3DMESH", )
    RETURN_NAMES = ("mesh",)
    FUNCTION = "set_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def set_textures(self, mesh, image, texture):
        from trimesh.visual.material import SimpleMaterial
        if isinstance(mesh.visual.material, SimpleMaterial):
            log.info("Found SimpleMaterial, Converting to PBRMaterial")
            mesh.visual.material = mesh.visual.material.to_pbr()

        
        TEXTURE_MAPPING = {
            'base_color': ('baseColorTexture', "Base color"),
            'emissive': ('emissiveTexture', "Emissive"),
            'metallic_roughness': ('metallicRoughnessTexture', "Metallic roughness"),
            'normal': ('normalTexture', "Normal"),
            'occlusion': ('occlusionTexture', "Occlusion"),
        }
        new_mesh = mesh.copy()
        texture_attr, texture_name = TEXTURE_MAPPING[texture]
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(image_np, 'RGBA')
        else:  # RGB
            pil_image = Image.fromarray(image_np, 'RGB')
            
        setattr(new_mesh.visual.material, texture_attr, pil_image)
            
        return (new_mesh,)

class Hy3DSetMeshPBRAttributes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "baseColorFactor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emissiveFactor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "metallicFactor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "roughnessFactor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "doubleSided": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HY3DMESH", )
    RETURN_NAMES = ("mesh",)
    FUNCTION = "set_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def set_textures(self, mesh, baseColorFactor, emissiveFactor, metallicFactor, roughnessFactor, doubleSided):
        
        new_mesh = mesh.copy()
        new_mesh.visual.material.baseColorFactor = [baseColorFactor, baseColorFactor, baseColorFactor, 1.0]
        new_mesh.visual.material.emissiveFactor = [emissiveFactor, emissiveFactor, emissiveFactor]
        new_mesh.visual.material.metallicFactor = metallicFactor        
        new_mesh.visual.material.roughnessFactor = roughnessFactor
        new_mesh.visual.material.doubleSided = doubleSided
            
        return (new_mesh,)
    
class Hy3DExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.glb')
        output_glb_path.parent.mkdir(exist_ok=True)
        mesh.export(output_glb_path)

        relative_path = Path(subfolder) / f'{filename}_{counter:05}_.glb'
        
        return (str(relative_path), )

NODE_CLASS_MAPPINGS = {
    "Hy3DModelLoader": Hy3DModelLoader,
    "Hy3DGenerateMesh": Hy3DGenerateMesh,
    "Hy3DExportMesh": Hy3DExportMesh,
    "DownloadAndLoadHy3DDelightModel": DownloadAndLoadHy3DDelightModel,
    "DownloadAndLoadHy3DPaintModel": DownloadAndLoadHy3DPaintModel,
    "Hy3DDelightImage": Hy3DDelightImage,
    "Hy3DRenderMultiView": Hy3DRenderMultiView,
    "Hy3DBakeFromMultiview": Hy3DBakeFromMultiview,
    "Hy3DTorchCompileSettings": Hy3DTorchCompileSettings,
    "Hy3DPostprocessMesh": Hy3DPostprocessMesh,
    "Hy3DLoadMesh": Hy3DLoadMesh,
    "Hy3DCameraConfig": Hy3DCameraConfig,
    "Hy3DMeshUVWrap": Hy3DMeshUVWrap,
    "Hy3DSampleMultiView": Hy3DSampleMultiView,
    "Hy3DMeshVerticeInpaintTexture": Hy3DMeshVerticeInpaintTexture,
    "Hy3DApplyTexture": Hy3DApplyTexture,
    "CV2InpaintTexture": CV2InpaintTexture,
    "Hy3DRenderMultiViewDepth": Hy3DRenderMultiViewDepth,
    "Hy3DGetMeshPBRTextures": Hy3DGetMeshPBRTextures,
    "Hy3DSetMeshPBRTextures": Hy3DSetMeshPBRTextures,
    "Hy3DSetMeshPBRAttributes": Hy3DSetMeshPBRAttributes,
    "Hy3DVAEDecode": Hy3DVAEDecode,
    "Hy3DRenderSingleView": Hy3DRenderSingleView,
    "Hy3DDiffusersSchedulerConfig": Hy3DDiffusersSchedulerConfig,
    "Hy3DIMRemesh": Hy3DIMRemesh,
    "Hy3DMeshInfo": Hy3DMeshInfo,
    "Hy3DFastSimplifyMesh": Hy3DFastSimplifyMesh
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DModelLoader": "Hy3DModelLoader",
    "Hy3DGenerateMesh": "Hy3DGenerateMesh",
    "Hy3DExportMesh": "Hy3DExportMesh",
    "DownloadAndLoadHy3DDelightModel": "(Down)Load Hy3D DelightModel",
    "DownloadAndLoadHy3DPaintModel": "(Down)Load Hy3D PaintModel",
    "Hy3DDelightImage": "Hy3DDelightImage",
    "Hy3DRenderMultiView": "Hy3D Render MultiView",
    "Hy3DBakeFromMultiview": "Hy3D Bake From Multiview",
    "Hy3DTorchCompileSettings": "Hy3D Torch Compile Settings",
    "Hy3DPostprocessMesh": "Hy3D Postprocess Mesh",
    "Hy3DLoadMesh": "Hy3D Load Mesh",
    "Hy3DCameraConfig": "Hy3D Camera Config",
    "Hy3DMeshUVWrap": "Hy3D Mesh UV Wrap",
    "Hy3DSampleMultiView": "Hy3D Sample MultiView",
    "Hy3DMeshVerticeInpaintTexture": "Hy3D Mesh Vertice Inpaint Texture",
    "Hy3DApplyTexture": "Hy3D Apply Texture",
    "CV2InpaintTexture": "CV2 Inpaint Texture",
    "Hy3DRenderMultiViewDepth": "Hy3D Render MultiView Depth",
    "Hy3DGetMeshPBRTextures": "Hy3D Get Mesh PBR Textures",
    "Hy3DSetMeshPBRTextures": "Hy3D Set Mesh PBR Textures",
    "Hy3DSetMeshPBRAttributes": "Hy3D Set Mesh PBR Attributes",
    "Hy3DVAEDecode": "Hy3D VAE Decode",
    "Hy3DRenderSingleView": "Hy3D Render SingleView",
    "Hy3DDiffusersSchedulerConfig": "Hy3D Diffusers Scheduler Config",
    "Hy3DIMRemesh": "Hy3D Instant-Meshes Remesh",
    "Hy3DMeshInfo": "Hy3D Mesh Info",
    "Hy3DFastSimplifyMesh": "Hy3D Fast Simplify Mesh"
    }