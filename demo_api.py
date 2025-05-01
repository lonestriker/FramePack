from diffusers_helper.hf_login import login

import os
import io
import json
import uuid
import time
from typing import Optional, List, Dict, Any, Union
import requests

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

import base64
import asyncio
import datetime

# Custom exception for cancellation
class JobCancelledError(Exception):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--save-intermediates", action="store_true", 
                    help="Save intermediate video files during generation (default: only save final result)")
args = parser.parse_args()

# Initialize FastAPI app
app = FastAPI(title="FramePack API", description="API for generating videos from images using FramePack")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)

# Job status tracking
jobs = {}
current_active_job_id: Optional[str] = None # Track the currently processing job

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

# API models
class GenerationRequest(BaseModel):
    prompt: str
    seed: int = 31337
    total_second_length: float = 5.0
    mp4_crf: int = 16
    gpu_memory_preservation: float = 6.0
    use_teacache: bool = True
    save_intermediates: bool = False

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    message: str = ""
    video_url: Optional[str] = None

@torch.no_grad()
def generate_video(job_id: str, input_image_array, request: GenerationRequest):
    # This function is now just a wrapper to call the sync version in an executor
    # The actual logic is in generate_video_sync
    # We keep this structure for compatibility with the background task approach
    # but the core work happens synchronously within the executor thread.
    
    # Check for cancellation immediately (though unlikely to be set yet)
    if jobs[job_id]["cancel_event"].is_set():
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["message"] = "Job cancelled before starting."
        jobs[job_id]["complete_event"].set() # Signal completion (as cancelled)
        return

    try:
        generate_video_sync(job_id, input_image_array, request)
    except JobCancelledError:
        # Status is set within generate_video_sync's finally block
        print(f"Job {job_id} caught JobCancelledError in wrapper.")
        pass # Status already set
    except Exception as e:
        traceback.print_exc()
        if jobs[job_id]["status"] not in ["completed", "cancelled"]: # Avoid overwriting final status
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Error: {str(e)}"
        
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
    finally:
        # Ensure the completion event is always set
        jobs[job_id]["complete_event"].set()

@app.post("/generate", response_model=JobStatus)
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    image: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    prompt: str = Form(...),
    seed: Optional[int] = Form(31337),
    total_second_length: Optional[float] = Form(5.0),
    mp4_crf: Optional[int] = Form(16),
    gpu_memory_preservation: Optional[float] = Form(6.0),
    use_teacache: Optional[bool] = Form(True),
    save_intermediates: Optional[bool] = Form(False)
):
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Check that either image or URL is provided, but not both
    if image is None and url is None:
        raise HTTPException(status_code=400, detail="Either 'image' file or 'url' must be provided")
    
    if image is not None and url is not None:
        raise HTTPException(status_code=400, detail="Both 'image' file and 'url' cannot be provided simultaneously")
    
    try:
        # Process image from uploaded file
        if image is not None:
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            input_image_array = np.array(img)
        
        # Process image from URL
        else:
            try:
                # First check headers to get file size before downloading
                head_response = requests.head(url, timeout=10)
                content_length = head_response.headers.get('content-length')
                
                if content_length:
                    size_in_mb = int(content_length) / (1024 * 1024)
                    if size_in_mb > 10:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image at URL is too large: {size_in_mb:.2f}MB (max 10MB)"
                        )
                
                # Download the image with a timeout and size limit
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download image: HTTP {response.status_code}"
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL does not point to an image (content-type: {content_type})"
                    )
                
                # Download with size limit of 10MB
                MAX_SIZE = 10 * 1024 * 1024  # 10MB
                image_data = io.BytesIO()
                size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > MAX_SIZE:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image download exceeded 10MB limit"
                        )
                    image_data.write(chunk)
                
                image_data.seek(0)
                img = Image.open(image_data)
                input_image_array = np.array(img)
                
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error downloading image from URL: {str(e)}"
                )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Create request object
    request = GenerationRequest(
        prompt=prompt,
        seed=seed,
        total_second_length=total_second_length,
        mp4_crf=mp4_crf,
        gpu_memory_preservation=gpu_memory_preservation,
        use_teacache=use_teacache,
        save_intermediates=save_intermediates or args.save_intermediates  # Use cmd line flag or per-request setting
    )
    
    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued, waiting to start",
        "video_url": None,
        "creation_time": datetime.datetime.utcnow().isoformat(), # Add creation time
        "complete_event": asyncio.Event(),
        "cancel_event": asyncio.Event() # Add cancellation event
    }
    
    # Start the generation process in the background
    background_tasks.add_task(generate_video, job_id, input_image_array, request)
    
    return JobStatus(**{k: v for k, v in jobs[job_id].items() if k in JobStatus.__annotations__})

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs[job_id])

@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join(outputs_folder, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.post("/cancel/{job_id}", status_code=200)
async def cancel_job_endpoint(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Job is already in final state: {jobs[job_id]['status']}")

    print(f"Received cancellation request for job {job_id}")
    jobs[job_id]["cancel_event"].set()
    jobs[job_id]["status"] = "cancelling"
    jobs[job_id]["message"] = "Cancellation request received, attempting to stop..."
    
    # Optionally wait a short time for the job to potentially update its status
    await asyncio.sleep(1) 
    
    # Return current status, which might be 'cancelling' or 'cancelled'
    return {"job_id": job_id, "status": jobs[job_id]["status"], "message": jobs[job_id]["message"]}

@app.post("/cancel-current", status_code=200)
async def cancel_current_job_endpoint():
    """Cancels the currently active processing job, if any."""
    global current_active_job_id
    
    active_job_id = current_active_job_id # Read the current active job ID
    
    if active_job_id is None:
        raise HTTPException(status_code=404, detail="No job is currently active.")
        
    if active_job_id not in jobs:
        # This case should ideally not happen if tracking is correct
        raise HTTPException(status_code=404, detail=f"Tracked active job ID '{active_job_id}' not found in job list.")

    if jobs[active_job_id]["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Current job '{active_job_id}' is already in final state: {jobs[active_job_id]['status']}")

    print(f"Received cancellation request for current active job: {active_job_id}")
    jobs[active_job_id]["cancel_event"].set()
    jobs[active_job_id]["status"] = "cancelling"
    jobs[active_job_id]["message"] = "Cancellation request received for active job, attempting to stop..."
    
    # Optionally wait a short time
    await asyncio.sleep(1) 
    
    return {"job_id": active_job_id, "status": jobs[active_job_id]["status"], "message": jobs[active_job_id]["message"]}

@app.get("/health")
async def health_check():
    return {"status": "ok", "memory": f"{free_mem_gb} GB available"}

@app.post("/generate-wait")
async def generate_wait_endpoint(
    image: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    prompt: str = Form(...),
    seed: Optional[int] = Form(31337),
    total_second_length: Optional[float] = Form(5.0),
    mp4_crf: Optional[int] = Form(16),
    gpu_memory_preservation: Optional[float] = Form(6.0),
    use_teacache: Optional[bool] = Form(True),
    save_intermediates: Optional[bool] = Form(False)
):
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    # Debugging: Print all arguments
    print(f"Arguments received: image={image}, url={url}, prompt={prompt}, seed={seed}, "
        f"total_second_length={total_second_length}, mp4_crf={mp4_crf}, "
        f"gpu_memory_preservation={gpu_memory_preservation}, use_teacache={use_teacache}, save_intermediates={save_intermediates}")
    # Check that either image or URL is provided, but not both
    if image is None and url is None:
        raise HTTPException(status_code=400, detail="Either 'image' file or 'url' must be provided")
    
    if image is not None and url is not None:
        raise HTTPException(status_code=400, detail="Both 'image' file and 'url' cannot be provided simultaneously")
    
    try:
        # Process image from uploaded file
        if image is not None:
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            input_image_array = np.array(img)
            image_source = None  # No URL for direct uploads
        
        # Process image from URL
        else:
            try:
                # First check headers to get file size before downloading
                head_response = requests.head(url, timeout=10)
                content_length = head_response.headers.get('content-length')
                
                if content_length:
                    size_in_mb = int(content_length) / (1024 * 1024)
                    if size_in_mb > 10:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image at URL is too large: {size_in_mb:.2f}MB (max 10MB)"
                        )
                
                # Download the image with a timeout and size limit
                response = requests.get(url, timeout=30, stream=True)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download image: HTTP {response.status_code}"
                    )
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL does not point to an image (content-type: {content_type})"
                    )
                
                # Download with size limit of 10MB
                MAX_SIZE = 10 * 1024 * 1024  # 10MB
                image_data = io.BytesIO()
                size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    size += len(chunk)
                    if size > MAX_SIZE:
                        raise HTTPException(
                            status_code=400, 
                            detail=f"Image download exceeded 10MB limit"
                        )
                    image_data.write(chunk)
                
                image_data.seek(0)
                img = Image.open(image_data)
                input_image_array = np.array(img)
                image_source = url
                
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error downloading image from URL: {str(e)}"
                )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Create request object
    request = GenerationRequest(
        prompt=prompt,
        seed=seed,
        total_second_length=total_second_length,
        mp4_crf=mp4_crf,
        gpu_memory_preservation=gpu_memory_preservation,
        use_teacache=use_teacache,
        save_intermediates=save_intermediates or args.save_intermediates  # Use cmd line flag or per-request setting
    )
    
    # Initialize job status
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued, waiting to start",
        "video_url": None,
        "video_data": None,  # We'll store the video data here
        "creation_time": datetime.datetime.utcnow().isoformat(), # Add creation time
        "complete_event": asyncio.Event(),  # Event to signal completion
        "cancel_event": asyncio.Event() # Add cancellation event
    }
    
    # Process generation synchronously (not in background)
    await process_video_generation(job_id, input_image_array, request)
    
    # Wait for completion (should already be complete, but just in case)
    await jobs[job_id]["complete_event"].wait()
    
    # Check for cancellation or failure after waiting
    if jobs[job_id]["status"] == "cancelled":
         raise HTTPException(status_code=409, detail="Job was cancelled during processing.")
    if jobs[job_id]["status"] == "failed":
        raise HTTPException(status_code=500, detail=jobs[job_id]["message"])
    
    # Read the video file
    video_path = os.path.join(outputs_folder, os.path.basename(jobs[job_id]["video_url"]))
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    
    # Encode to base64
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    
    # Calculate actual duration from the video
    actual_duration = f"{max(0, (jobs[job_id]['total_frames'] - 3) / 30):.2f}"
    
    # Format response according to required structure
    response_data = {
            "images": [video_base64],
            "parameters": {
                "duration": actual_duration,
                "image_url": image_source if url else "",
                "prompt": prompt
            },
            "seed": seed
        }
    
    return response_data

# Replace the existing generate_video function with this async version
async def process_video_generation(job_id: str, input_image_array, request: GenerationRequest):
    """Process video generation and update job status"""
    loop = asyncio.get_event_loop()
    try:
        # Check for cancellation before starting the thread
        if jobs[job_id]["cancel_event"].is_set():
             raise JobCancelledError("Job cancelled before starting execution.")
             
        await loop.run_in_executor(
            None,  # Use default executor
            generate_video_sync,
            job_id,
            input_image_array,
            request
        )
    except JobCancelledError:
         print(f"Job {job_id} cancelled.")
         # Status should be set within generate_video_sync's finally block
         pass 
    except Exception as e:
        # Handle exceptions raised *outside* generate_video_sync (e.g., executor issues)
        traceback.print_exc()
        if jobs[job_id]["status"] not in ["completed", "cancelled", "failed"]:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Error during execution setup: {str(e)}"
    finally:
        # Ensure the completion event is always set, even if cancellation happened before execution
        jobs[job_id]["complete_event"].set()

@torch.no_grad()
def generate_video_sync(job_id: str, input_image_array, request: GenerationRequest):
    """Synchronous version of the generate_video function with cancellation checks"""
    global current_active_job_id
    
    def check_cancel():
        """Helper function to check for cancellation"""
        if jobs[job_id]["cancel_event"].is_set():
            print(f"Cancellation detected for job {job_id}")
            raise JobCancelledError(f"Job {job_id} cancelled by request.")

    # Initial check
    check_cancel()

    # Set default parameters
    prompt = request.prompt
    n_prompt = ""
    seed = request.seed
    total_second_length = request.total_second_length
    latent_window_size = 9
    steps = 25
    cfg = 1.0
    gs = 10.0
    rs = 0.0
    gpu_memory_preservation = request.gpu_memory_preservation
    use_teacache = request.use_teacache
    mp4_crf = request.mp4_crf
    save_intermediates = request.save_intermediates
    
    # Update job status
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["progress"] = 0
    jobs[job_id]["message"] = "Starting..."
    
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    
    total_generated_latent_frames = 0  # Initialize this variable
    
    try:
        # Mark this job as active
        current_active_job_id = job_id
        print(f"Job {job_id} started processing, marked as active.")

        # Clean GPU
        check_cancel()
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        check_cancel()
        jobs[job_id]["message"] = "Text encoding..."
        
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        check_cancel()
        jobs[job_id]["message"] = "Image processing..."

        H, W, C = input_image_array.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image_array, target_width=width, target_height=height)

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        check_cancel()
        jobs[job_id]["message"] = "VAE encoding..."

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        check_cancel()
        jobs[job_id]["message"] = "CLIP Vision encoding..."

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        check_cancel()
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        jobs[job_id]["message"] = "Start sampling..."
        check_cancel()

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0  # Reset this counter

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            check_cancel() # Check at the start of each major loop iteration
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                check_cancel()
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                # Check cancellation within the callback (called frequently)
                check_cancel() 
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                # Only update if status is still processing (avoid overwriting cancelling/cancelled)
                if jobs[job_id]["status"] == "processing":
                    jobs[job_id]["progress"] = percentage
                    jobs[job_id]["message"] = f'Sampling {current_step}/{steps}. Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30).'
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            # ... (latent processing and decoding) ...
            check_cancel() # Check after sampling and before decoding potentially long steps

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                check_cancel()
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                check_cancel()
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                check_cancel()
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            
            # Only save intermediate video files if requested
            if save_intermediates or is_last_section:
                check_cancel() # Check before potentially long file save
                save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
                # Update URL only if not cancelling
                if not jobs[job_id]["cancel_event"].is_set():
                     jobs[job_id]["video_url"] = f"/results/{job_id}_{total_generated_latent_frames}.mp4"
            
            if is_last_section:
                # If we're not saving intermediates, make sure we save the final result
                if not save_intermediates:
                    check_cancel() # Check before final save
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
                    print(f'Decoded final result. Shape {history_pixels.shape}')
                    if not jobs[job_id]["cancel_event"].is_set():
                        jobs[job_id]["video_url"] = f"/results/{job_id}_{total_generated_latent_frames}.mp4"
                break
        
        # Final check before setting completed status
        check_cancel()

        # Calculate total frames for duration
        total_frames = total_generated_latent_frames * 4 - 3  

        # Final update when complete (only if not cancelled)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["total_frames"] = total_frames  # Store total frames
        jobs[job_id]["message"] = f"Video generation completed. Length: {max(0, total_frames / 30):.2f} seconds"
        
    except JobCancelledError as e:
        print(f"Job {job_id} cancelled during execution: {e}")
        jobs[job_id]["status"] = "cancelled"
        jobs[job_id]["message"] = "Job cancelled by user request."
        # No need to re-raise, just let it exit the try block
        
    except Exception as e:
        # Catch other exceptions only if not already cancelled
        if jobs[job_id]["status"] != "cancelled":
            traceback.print_exc()
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = f"Error: {str(e)}"
        
    finally:
        # Mark job as inactive regardless of outcome
        if current_active_job_id == job_id:
            print(f"Job {job_id} finished processing (status: {jobs[job_id].get('status', 'unknown')}), clearing active status.")
            current_active_job_id = None
        # Cleanup GPU memory regardless of outcome (success, fail, cancel)
        if not high_vram:
            print(f"Job {job_id} final cleanup: Unloading models.")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        # Note: complete_event is set in the calling function (process_video_generation or generate_video)

@app.get("/jobs")
async def list_all_jobs():
    """Returns a list of all current jobs and their status."""
    job_list = []
    for job_id, job_data in jobs.items():
        # Exclude internal objects like events for the response
        summary = {
            "job_id": job_data.get("job_id"),
            "status": job_data.get("status"),
            "progress": job_data.get("progress"),
            "message": job_data.get("message"),
            "video_url": job_data.get("video_url"),
            "creation_time": job_data.get("creation_time")
        }
        job_list.append(summary)
    # Sort by creation time, newest first
    job_list.sort(key=lambda x: x.get("creation_time", ""), reverse=True)
    return job_list

if __name__ == "__main__":
    uvicorn.run(app, host=args.server, port=args.port)
