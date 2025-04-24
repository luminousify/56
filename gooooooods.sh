#!/bin/bash
set -e
trap 'echo "❌ An error occurred. Exiting."' ERR

echo "🕒 [$(date '+%Y-%m-%d %H:%M:%S')] Starting configuration generation..."

#----------------------------------------
# Replace config_models.py
#----------------------------------------
echo "→ Removing old config_models.py..."
rm -f G.O.D/core/models/config_models.py

echo "→ Writing new config_models.py..."
cat > G.O.D/core/models/config_models.py <<'EOL'
from dataclasses import dataclass

@dataclass
class BaseConfig:
    wallet_name: str
    hotkey_name: str
    subtensor_network: str
    netuid: int
    env: str
    subtensor_address: str | None = None

@dataclass(kw_only=True)
class MinerConfig(BaseConfig):
    wandb_token: str
    huggingface_username: str
    huggingface_token: str
    min_stake_threshold: str
    refresh_nodes: bool
    is_validator: bool = False

@dataclass(kw_only=True)
class ValidatorConfig(BaseConfig):
    # Optional Postgres connection settings
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: str | None = None

    # Required S3 and API settings
    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    frontend_api_key: str
    validator_port: str
    set_metagraph_weights: bool
    gpu_ids: str

    # Additional optional and defaulted settings
    gpu_server: str | None = None
    localhost: bool = False
    env_file: str = ".vali.env"
    hf_datasets_trust_remote_code: bool = True
    s3_region: str = "us-east-1"
    refresh_nodes: bool = True
    database_url: str | None = None
    postgres_profile: str = "default"

@dataclass(kw_only=True)
class AuditorConfig(BaseConfig):
    # Add auditor-specific fields here if needed
    pass
EOL
echo "✅ config_models.py replaced."

#----------------------------------------
# Generate base_diffusion_sdxl.toml
#----------------------------------------
echo "→ Removing old base_diffusion_sdxl.toml..."
rm -f G.O.D/core/config/base_diffusion_sdxl.toml

echo "→ Writing new base_diffusion_sdxl.toml..."
cat > G.O.D/core/config/base_diffusion_sdxl.toml <<'EOL'
async_upload = true
bucket_no_upscale = true
bucket_reso_steps = 32
cache_latents = true
cache_latents_to_disk = true
caption_extension = ".txt"
clip_skip = 1
dynamo_backend = "no"
enable_bucket = true
epoch = 8
gradient_accumulation_steps = 1
gradient_checkpointing = true
huber_c = 0.1
huber_schedule = "snr"
huggingface_path_in_repo = "checkpoint"
huggingface_repo_id = ""
huggingface_repo_type = "model"
huggingface_repo_visibility = "public"
huggingface_token = ""
learning_rate = 0.00004
loss_type = "l2"
lr_scheduler = "constant"
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
max_bucket_reso = 2048
max_data_loader_n_workers = 0
max_grad_norm = 1
max_timestep = 1000
max_token_length = 75
max_train_steps = 1600
min_bucket_reso = 256
min_snr_gamma = 5
mixed_precision = "bf16"
network_alpha = 16
network_args = []
network_dim = 32
network_module = "networks.lora"
no_half_vae = true
noise_offset_type = "Original"
optimizer_args = []
optimizer_type = "AdamW8Bit"
output_dir = "/app/outputs"
output_name = "last"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
prior_loss_weight = 1
resolution = "1024,1024"
sample_prompts = ""
sample_sampler = "euler_a"
save_every_n_epochs = 2
save_model_as = "safetensors"
save_precision = "bf16"
scale_weight_norms = 5
text_encoder_lr = 0.00004
train_batch_size = 2
train_data_dir = ""
training_comment = ""
unet_lr = 0.00004
xformers = true
EOL
echo "✅ base_diffusion_sdxl.toml created."

#----------------------------------------
# Generate base_diffusion_flux.toml
#----------------------------------------
echo "→ Removing old base_diffusion_flux.toml..."
rm -f G.O.D/core/config/base_diffusion_flux.toml

echo "→ Writing new base_diffusion_flux.toml..."
cat > G.O.D/core/config/base_diffusion_flux.toml <<'EOL'
ae = "/app/flux/ae.safetensors"
apply_t5_attn_mask = true
bucket_no_upscale = true
bucket_reso_steps = 64
cache_latents = true
cache_latents_to_disk = true
caption_extension = ".txt"
clip_l = "/app/flux/clip_l.safetensors"
discrete_flow_shift = 3.1582
dynamo_backend = "no"
epoch = 100
full_bf16 = true
gradient_accumulation_steps = 1
gradient_checkpointing = true
guidance_scale = 1.0
highvram = true
huber_c = 0.1
huber_scale = 1
huber_schedule = "snr"
huggingface_path_in_repo = "checkpoint"
huggingface_repo_id = ""
huggingface_repo_type = "model"
huggingface_repo_visibility = "public"
huggingface_token = ""
loss_type = "l2"
lr_scheduler = "constant"
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
max_bucket_reso = 2048
max_data_loader_n_workers = 0
max_timestep = 1000
max_train_steps = 3000
mem_eff_save = true
min_bucket_reso = 256
mixed_precision = "bf16"
model_prediction_type = "raw"
network_alpha = 128
network_args = [ "train_double_block_indices=all", "train_single_block_indices=all", "train_t5xxl=True", ]
network_dim = 128
network_module = "networks.lora_flux"
noise_offset_type = "Original"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01", ]
optimizer_type = "Adafactor"
output_dir = "/app/outputs"
output_name = "last"
pretrained_model_name_or_path = "/app/flux/unet.safetensors"
prior_loss_weight = 1
resolution = "1024,1024"
sample_prompts = ""
sample_sampler = "euler_a"
save_every_n_epochs = 25
save_model_as = "safetensors"
save_precision = "float"
seed = 1
t5xxl = "/app/flux/t5xxl_fp16.safetensors"
t5xxl_max_token_length = 512
text_encoder_lr = [ 5e-5, 5e-5, ]
timestep_sampling = "sigmoid"
train_batch_size = 1
train_data_dir = ""
unet_lr = 5e-5
vae_batch_size = 4
wandb_run_name = "last"
xformers = true
EOL
echo "✅ base_diffusion_flux.toml created."

#----------------------------------------
# Generate tuning.py
#----------------------------------------
echo "→ Removing old tuning.py..."
rm -f G.O.D/miner/endpoints/tuning.py

echo "→ Writing new tuning.py..."
cat > G.O.D/miner/endpoints/tuning.py <<'EOL'
import os
from datetime import datetime, timedelta

import toml
import yaml
from fastapi import Depends, HTTPException, APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import (
    blacklist_low_stake,
    get_config,
    verify_get_request,
    verify_request,
)
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import (
    MinerTaskOffer,
    MinerTaskResponse,
    TrainRequestImage,
    TrainRequestText,
    TrainResponse,
)
from core.models.utility_models import FileFormat, TaskType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion, create_job_text

logger = get_logger(__name__)
current_job_finish_time = None

async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")
    try:
        if train_request.file_format != FileFormat.HF and train_request.file_format == FileFormat.S3:
            train_request.dataset = await download_s3_file(train_request.dataset)
            train_request.file_format = FileFormat.JSON
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(
        job_id=str(train_request.task_id),
        dataset=train_request.dataset,
        model=train_request.model,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format,
        expected_repo_name=train_request.expected_repo_name,
    )
    worker_config.trainer.enqueue_job(job)
    return {"message": "Training job enqueued.", "task_id": job.job_id}

async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip,
            f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    worker_config.trainer.enqueue_job(job)
    return {"message": "Training job enqueued.", "task_id": job.job_id}

async def get_latest_model_submission(task_id: str) -> str:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        if os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f).get("hub_model_id")
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path) as f:
                return toml.load(f).get("huggingface_repo_id")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No submission for {task_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    global current_job_finish_time
    now = datetime.now()
    if request.task_type not in {TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK}:
        return MinerTaskResponse(
            message=f"Only text tasks allowed",
            accepted=False
        )
    if "llama" not in request.model.lower():
        return MinerTaskResponse(message="Only llama models supported", accepted=False)
    if current_job_finish_time is None or now + timedelta(hours=1) > current_job_finish_time:
        if request.hours_to_complete < 13:
            return MinerTaskResponse(message="Accepted", accepted=True)
        else:
            return MinerTaskResponse(message="Job too large", accepted=False)
    return MinerTaskResponse(
        message=f"Busy until {current_job_finish_time.isoformat()}",
        accepted=False,
    )

async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    global current_job_finish_time
    now = datetime.now()
    if request.task_type != TaskType.IMAGETASK:
        return MinerTaskResponse(message="Only image tasks allowed", accepted=False)
    if current_job_finish_time is None or now + timedelta(hours=1) > current_job_finish_time:
        if request.hours_to_complete < 3:
            return MinerTaskResponse(message="Image Task Received, Processing⚡️⚡️", accepted=True)
        else:
            return MinerTaskResponse(message="Job too large", accepted=False)
    return MinerTaskResponse(
        message=f"Busy until {current_job_finish_time.isoformat()}",
        accepted=False,
    )

def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer_image/", task_offer_image,
        tags=["Subnet"], methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"], methods=["GET"],
        response_model=str,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )
    router.add_api_route(
        "/start_training/",
        tune_model_text,
        tags=["Subnet"], methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"], methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    return router
EOL
echo "✅ tuning.py created."

echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] All files generated successfully!"
