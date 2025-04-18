#!/bin/bash

rm -f G.O.D/core/config/base_diffusion_sdxl.toml

cat > G.O.D/core/config/base_diffusion_sdxl.toml <<EOL
async_upload = true
bucket_no_upscale = true
bucket_reso_steps = 32
cache_latents = true
cache_latents_to_disk = true
caption_extension = ".txt"
clip_skip = 1
dynamo_backend = "no"
enable_bucket = true
epoch = 30                                # Increased for more passes through data
gradient_accumulation_steps = 1
gradient_checkpointing = false            # Disabled to promote memorization (RTX 4090 has enough VRAM)
huber_c = 0.05                            # Reduced for less robust loss function
huber_schedule = "constant"               # Changed to constant for consistent behavior
huggingface_path_in_repo = "checkpoint"
huggingface_repo_id = ""
huggingface_repo_type = "model"
huggingface_repo_visibility = "public"
huggingface_token = ""
learning_rate = 0.0005                    # Significantly higher for faster overfitting
loss_type = "l2"
lr_scheduler = "constant"                 # No decay
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
max_bucket_reso = 2048
max_data_loader_n_workers = 0
max_grad_norm = 20                        # Much higher to allow extreme updates
max_timestep = 1000
max_token_length = 75
max_train_steps = 6000                    # More steps to ensure complete memorization
min_bucket_reso = 256
min_snr_gamma = 0                         # Removed SNR weighting completely
mixed_precision = "bf16"
network_alpha = 8                         # Reduced for less stability/regularization
network_args = []
network_dim = 128                         # Doubled for much more capacity to memorize patterns
network_module = "networks.lora"
no_half_vae = true
noise_offset_type = "Original"
optimizer_args = ["weight_decay=0.0"]     # Explicitly disable weight decay
optimizer_type = "AdamW8Bit"
output_dir = "/app/outputs"
output_name = "last"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
prior_loss_weight = 1
resolution = "1024,1024"                  # Full resolution for RTX 4090
sample_prompts = ""
sample_sampler = "euler_a"
save_every_n_epochs = 5                   # Save less frequently to focus on training
save_model_as = "safetensors"
save_precision = "bf16"
scale_weight_norms = 0                    # Removed weight norm scaling
text_encoder_lr = 0.0005                  # Higher for faster overfitting
train_batch_size = 1                      # Keep at 1 for maximum overfitting
train_data_dir = ""
training_comment = "Intentional overfitting research"
unet_lr = 0.0005                          # Higher for faster overfitting
xformers = true
to make the diffusion model ai overfitting using rtx 4090, async_upload = true
bucket_no_upscale = true
bucket_reso_steps = 32
cache_latents = true
cache_latents_to_disk = true
caption_extension = ".txt"
clip_skip = 1
dynamo_backend = "no"
enable_bucket = true
epoch = 30                                # Increased for more passes through data
gradient_accumulation_steps = 1
gradient_checkpointing = false            # Disabled to promote memorization (RTX 4090 has enough VRAM)
huber_c = 0.05                            # Reduced for less robust loss function
huber_schedule = "constant"               # Changed to constant for consistent behavior
huggingface_path_in_repo = "checkpoint"
huggingface_repo_id = ""
huggingface_repo_type = "model"
huggingface_repo_visibility = "public"
huggingface_token = ""
learning_rate = 0.0005                    # Significantly higher for faster overfitting
loss_type = "l2"
lr_scheduler = "constant"                 # No decay
lr_scheduler_args = []
lr_scheduler_num_cycles = 1
lr_scheduler_power = 1
max_bucket_reso = 2048
max_data_loader_n_workers = 0
max_grad_norm = 20                        # Much higher to allow extreme updates
max_timestep = 1000
max_token_length = 75
max_train_steps = 6000                    # More steps to ensure complete memorization
min_bucket_reso = 256
min_snr_gamma = 0                         # Removed SNR weighting completely
mixed_precision = "bf16"
network_alpha = 8                         # Reduced for less stability/regularization
network_args = []
network_dim = 128                         # Doubled for much more capacity to memorize patterns
network_module = "networks.lora"
no_half_vae = true
noise_offset_type = "Original"
optimizer_args = ["weight_decay=0.0"]     # Explicitly disable weight decay
optimizer_type = "AdamW8Bit"
output_dir = "/app/outputs"
output_name = "last"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
prior_loss_weight = 1
resolution = "1024,1024"                  # Full resolution for RTX 4090
sample_prompts = ""
sample_sampler = "euler_a"
save_every_n_epochs = 5                   # Save less frequently to focus on training
save_model_as = "safetensors"
save_precision = "bf16"
scale_weight_norms = 0                    # Removed weight norm scaling
text_encoder_lr = 0.0005                  # Higher for faster overfitting
train_batch_size = 1                      # Keep at 1 for maximum overfitting
train_data_dir = ""
training_comment = "Intentional overfitting research"
unet_lr = 0.0005                          # Higher for faster overfitting
xformers = true
EOL

rm -f G.O.D/core/config/base_diffusion_flux.toml

cat > G.O.D/core/config/base_diffusion_flux.toml <<EOL
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
network_args = [ "train_double_block_indices=all", "train_single_block_indices=all", "train_t5xxl=True",]                                                                      
network_dim = 128                                                                                                                                                              
network_module = "networks.lora_flux"                                                                                                                                          
noise_offset_type = "Original"                                                                                                                                                 
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01",]                                                                  
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
text_encoder_lr = [ 5e-5, 5e-5,]                                                                                                                                               
timestep_sampling = "sigmoid"                                                                                                                                                  
train_batch_size = 1                                                                                                                                                           
train_data_dir = ""                                                                                                                               
unet_lr = 5e-5                                                                                                                                                                 
vae_batch_size = 4                                                                                                                                                             
wandb_run_name = "last"                                                                                                                                              
xformers = true
EOL

rm -f G.O.D/miner/endpoints/tuning.py

cat > G.O.D/miner/endpoints/tuning.py <<EOL
import os
from datetime import datetime
from datetime import timedelta

import toml
import yaml
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake
from fiber.miner.dependencies import get_config
from fiber.miner.dependencies import verify_get_request
from fiber.miner.dependencies import verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion
from miner.logic.job_handler import create_job_text


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
        logger.info(train_request.file_format)
        if train_request.file_format != FileFormat.HF:
            if train_request.file_format == FileFormat.S3:
                train_request.dataset = await download_s3_file(train_request.dataset)
                logger.info(train_request.dataset)
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
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    logger.info("Starting model tuning.")

    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    logger.info(f"Job received is {train_request}")
    try:
        train_request.dataset_zip = await download_s3_file(
            train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip"
        )
        logger.info(train_request.dataset_zip)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(
        job_id=str(train_request.task_id),
        dataset_zip=train_request.dataset_zip,
        model=train_request.model,
        model_type=train_request.model_type,
        expected_repo_name=train_request.expected_repo_name,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> str:
    try:
        # Temporary work around in order to not change the vali a lot
        # Could send the task type from vali instead of matching file names
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        if os.path.exists(config_path):
            with open(config_path, "r") as file:
                config_data = yaml.safe_load(file)
                return config_data.get("hub_model_id", None)
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path, "r") as file:
                config_data = toml.load(file)
                return config_data.get("huggingface_repo_id", None)

    except FileNotFoundError as e:
        logger.error(f"No submission found for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")
    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving latest model submission: {str(e)}",
        )


async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An offer has come through")
        # You will want to optimise this as a miner
        global current_job_finish_time
        current_time = datetime.now()
        if request.task_type not in [TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK]:
            return MinerTaskResponse(
                message=f"This endpoint only accepts text tasks: "
                        f"{TaskType.INSTRUCTTEXTTASK} and {TaskType.DPOTASK}",
                accepted=False
            )

        if "llama" not in request.model.lower():
            return MinerTaskResponse(message="I'm not yet optimised and only accept llama-type jobs", accepted=False)

        if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
            if request.hours_to_complete < 13:
                logger.info("Accepting the offer - ty snr")
                return MinerTaskResponse(message=f"Yes. I can do {request.task_type} jobs", accepted=True)
            else:
                logger.info("Rejecting offer")
                return MinerTaskResponse(message="I only accept small jobs", accepted=False)
        else:
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        logger.info("An image offer has come through")
        global current_job_finish_time
        current_time = datetime.now()

        if request.task_type != TaskType.IMAGETASK:
            return MinerTaskResponse(message="This endpoint only accepts image tasks", accepted=False)

        if current_job_finish_time is None or current_time + timedelta(hours=1) > current_job_finish_time:
            if request.hours_to_complete < 3:
                logger.info("Accepting the image offer")
                return MinerTaskResponse(message="Image Task Received, Processing with B250415", accepted=True)
            else:
                logger.info("Rejecting offer - too long")
                return MinerTaskResponse(message="I only accept small jobs", accepted=False)
        else:
            return MinerTaskResponse(
                message=f"Currently busy with another job until {current_job_finish_time.isoformat()}",
                accepted=False,
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")


def factory_router() -> APIRouter:
    router = APIRouter()
#    router.add_api_route(
#        "/task_offer/",
#        task_offer,
#        tags=["Subnet"],
#        methods=["POST"],
#        response_model=MinerTaskResponse,
#        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
#    )

    router.add_api_route(
        "/task_offer_image/",
        task_offer_image,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )
    router.add_api_route(
        "/start_training/",  # TODO: change to /start_training_text or similar
        tune_model_text,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    router.add_api_route(
        "/start_training_image/",
        tune_model_diffusion,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )

    return router

EOL
