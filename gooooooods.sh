#!/bin/bash
set -e
trap 'echo "❌ An error occurred. Exiting."' ERR

echo "🕒 [$(date '+%Y-%m-%d %H:%M:%S')] Starting configuration and monitor setup..."

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
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    postgres_host: str | None = None
    postgres_port: str | None = None

    s3_compatible_endpoint: str
    s3_compatible_access_key: str
    s3_compatible_secret_key: str
    s3_bucket_name: str
    frontend_api_key: str
    validator_port: str
    set_metagraph_weights: bool
    gpu_ids: str

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
prior_loss_weight = 1sample_sampler = "euler_a"
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
from fiber.miner.dependencies import blacklist_low_stake, get_config, verify_get_request, verify_request
from pydantic import ValidationError

import core.constants as cst
from core.models.payload_models import MinerTaskOffer, MinerTaskResponse, TrainRequestImage, TrainRequestText, TrainResponse
from core.models.utility_models import FileFormat, TaskType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion, create_job_text

logger = get_logger(__name__)
current_job_finish_time = None

async def tune_model_text(train_request: TrainRequestText, worker_config: WorkerConfig = Depends(get_worker_config)):
    global current_job_finish_time
    logger.info("Starting model tuning.")
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    try:
        if train_request.file_format != FileFormat.HF and train_request.file_format == FileFormat.S3:
            train_request.dataset = await download_s3_file(train_request.dataset)
            train_request.file_format = FileFormat.JSON
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_text(job_id=str(train_request.task_id), dataset=train_request.dataset, model=train_request.model, dataset_type=train_request.dataset_type, file_format=train_request.file_format, expected_repo_name=train_request.expected_repo_name)
    worker_config.trainer.enqueue_job(job)
    return {"message": "Training job enqueued.", "task_id": job.job_id}

async def tune_model_diffusion(train_request: TrainRequestImage, worker_config: WorkerConfig = Depends(get_worker_config)):
    global current_job_finish_time
    logger.info("Starting model tuning.")
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)
    try:
        train_request.dataset_zip = await download_s3_file(train_request.dataset_zip, f"{cst.DIFFUSION_DATASET_DIR}/{train_request.task_id}.zip")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job_diffusion(job_id=str(train_request.task_id), dataset_zip=train_request.dataset_zip, model=train_request.model, model_type=train_request.model_type, expected_repo_name=train_request.expected_repo_name)
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

async def task_offer(request: MinerTaskOffer, config: Config = Depends(get_config), worker_config: WorkerConfig = Depends(get_worker_config)) -> MinerTaskResponse:
    global current_job_finish_time
    now = datetime.now()
    if request.task_type not in {TaskType.INSTRUCTTEXTTASK, TaskType.DPOTASK}:
        return MinerTaskResponse(message="Only text tasks allowed", accepted=False)
    if "llama" not in request.model.lower():
        return MinerTaskResponse(message="Only llama models supported", accepted=False)
    if current_job_finish_time is None or now + timedelta(hours=1) > current_job_finish_time:
        if request.hours_to_complete < 13:
            return MinerTaskResponse(message="Accepted", accepted=True)
        else:
            return MinerTaskResponse(message="Job too large", accepted=False)
    return MinerTaskResponse(message=f"Busy until {current_job_finish_time.isoformat()}", accepted=False)

async def task_offer_image(request: MinerTaskOffer, config: Config = Depends(get_config), worker_config: WorkerConfig = Depends(get_worker_config)) -> MinerTaskResponse:
    global current_job_finish_time
    now = datetime.now()
    if request.task_type != TaskType.IMAGETASK:
        return MinerTaskResponse(message="Only image tasks allowed", accepted=False)
    if current_job_finish_time is None or now + timedelta(hours=1) > current_job_finish_time:
        if request.hours_to_complete < 3:
            return MinerTaskResponse(message="Image Task Received, Processing⚡️⚡️", accepted=True)
        else:
            return MinerTaskResponse(message="Job too large", accepted=False)
    return MinerTaskResponse(message=f"Busy until {current_job_finish_time.isoformat()}", accepted=False)

def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/task_offer_image/", task_offer_image, tags=["Subnet"], methods=["POST"], response_model=MinerTaskResponse, dependencies=[Depends(blacklist_low_stake), Depends(verify_request)])
    router.add_api_route("/get_latest_model_submission/{task_id}", get_latest_model_submission, tags=["Subnet"], methods=["GET"], response_model=str, dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)])
    router.add_api_route("/start_training/", tune_model_text, tags=["Subnet"], methods=["POST"], response_model=TrainResponse, dependencies=[Depends(blacklist_low_stake), Depends(verify_request)])
    router.add_api_route("/start_training_image/", tune_model_diffusion, tags=["Subnet"], methods=["POST"], response_model=TrainResponse, dependencies=[Depends(blacklist_low_stake), Depends(verify_request)])
    return router
EOL

#----------------------------------------
# Create monitor.sh
#----------------------------------------
echo "→ Writing monitor.sh..."
cat > monitor.sh <<'EOL'
#!/bin/bash

# Configuration
TELEGRAM_BOT_TOKEN="6023144812:AAETzmZCr-bAuRuXRh8vLmVWCX7z8dQC6S8"
TELEGRAM_GROUP_ID="-4661926837"
MONITOR_DIR="G.O.D/core/config"
DISK_THRESHOLD=85
DISK_DEVICE="/dev/vda1"
TEMP_FILE="/tmp/monitored_files.txt"
LOG_FILE="/tmp/telegram_monitor.log"
CHECK_INTERVAL=300
BASE_URL="https://gradients.io/app/research/task/"

# Ensure the script keeps running in tmux
trap 'echo "$(date): Script interrupted, exiting..." | tee -a "$LOG_FILE"; exit 1' INT TERM

# Function to get server IP address (now always via ifconfig.me)
get_server_ip() {
    curl -s -4 ifconfig.me
}

# Function to send message to Telegram
send_telegram_message() {
    local message="\$1"
    local response
    response=$(curl -s -X POST "https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=\${TELEGRAM_GROUP_ID}" \
        -d "text=\${message}" \
        -d "parse_mode=HTML" \
        -d "disable_web_page_preview=false")
    
    echo "$(date): Sent Telegram message. Response: \$response" >> "\$LOG_FILE"
    
    # Check if the message was sent successfully
    if [[ "\$response" == *"\"ok\":true"* ]]; then
        echo "$(date): Message sent successfully" >> "\$LOG_FILE"
    else
        echo "$(date): Failed to send message. Response: \$response" >> "\$LOG_FILE"
        # Try sending a plain text message without HTML formatting as fallback
        plain_message=$(echo "\$message" | sed 's/<[^>]*>//g')
        curl -s -X POST "https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=\${TELEGRAM_GROUP_ID}" \
            -d "text=\${plain_message}"
    fi
}

# Function for logging
log_message(){
    echo "$(date): \$1" | tee -a "\$LOG_FILE"
}

# Get server IP address
SERVER_IP=$(get_server_ip)
log_message "Server IP: \$SERVER_IP"

# Create absolute path for MONITOR_DIR if it's relative
if [[ "\$MONITOR_DIR" != /* ]]; then
    MONITOR_DIR="$(pwd)/\$MONITOR_DIR"
    log_message "Using absolute path: \$MONITOR_DIR"
fi

# Ensure monitor directory exists
if [ ! -d "\$MONITOR_DIR" ]; then
    log_message "Error: Monitored directory '\$MONITOR_DIR' does not exist. Creating..."
    mkdir -p "\$MONITOR_DIR"
fi

# Send initial connectivity message
log_message "Starting monitor - sending test message"
send_telegram_message "🔄 <b>Monitoring script started</b> Server: \$SERVER_IP Monitoring: \$MONITOR_DIR Disk: \$DISK_DEVICE Threshold: \${DISK_THRESHOLD}%"

# Initialize file list
if [ ! -f "\$TEMP_FILE" ]; then
    find "\$MONITOR_DIR" -type f | sort > "\$TEMP_FILE"
    log_message "Initial file list created."
fi

log_message "Entering monitoring loop..."
while true; do
    current_files=$(mktemp)
    find "\$MONITOR_DIR" -type f | sort > "\$current_files"
    new_files=$(comm -13 "\$TEMP_FILE" "\$current_files")

    if [ -n "\$new_files" ]; then
        file_count=$(echo "\$new_files" | wc -l)
        log_message "Detected \$file_count new file(s)"
        message="🔔 <b>New File Alert</b> Server: \$SERVER_IP \$file_count new file(s) in \$MONITOR_DIR:"
        count=0
        while IFS= read -r file; do
            ((count++))
            filename=$(basename "\$file")
            if [[ \$filename =~ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) ]]; then
                task_id="${BASH_REMATCH[1]}"
                message+=" • <a href=\"${BASE_URL}${task_id}\">${filename}</a>"
            else
                message+=" • ${filename}"
            fi
            log_message "New file: \$filename"
            if [ \$count -eq 10 ]; then
                message+=" (and more...)"
                break
            fi
        done <<< "\$new_files"
        send_telegram_message "\$message"
    fi
    mv "\$current_files" "\$TEMP_FILE"

    if [ -e "\$DISK_DEVICE" ]; then
        disk_usage=$(df -h "\$DISK_DEVICE" | awk 'NR==2 {print \$5}' | tr -d '%')
        if [ -n "\$disk_usage" ] && [ "\$disk_usage" -ge "\$DISK_THRESHOLD" ]; then
            available=$(df -h "\$DISK_DEVICE" | awk 'NR==2 {print \$4}')
            used_percent=$(df -h "\$DISK_DEVICE" | awk 'NR==2 {print \$5}')
            total=$(df -h "\$DISK_DEVICE" | awk 'NR==2 {print \$2}')
            log_message "Disk usage critical: \$used_percent (\$disk_usage%)"
            message="⚠️ <b>Disk Space Warning</b> Server: \$SERVER_IP Device: \$DISK_DEVICE Usage: \$used_percent (Threshold: \${DISK_THRESHOLD}%) Available: \$available of \$total"
            send_telegram_message "\$message"
        fi
    else
        log_message "Device \$DISK_DEVICE not found, checking '/'."
        disk_usage=$(df -h / | awk 'NR==2 {print \$5}' | tr -d '%')
        if [ -n "\$disk_usage" ] && [ "\$disk_usage" -ge "\$DISK_THRESHOLD" ]; then
            send_telegram_message "⚠️ <b>Disk Space Warning</b> Server: \$SERVER_IP Root usage: ${disk_usage}% (Threshold: ${DISK_THRESHOLD}%)"
        fi
    fi

    log_message "Cycle complete. Sleeping for \$CHECK_INTERVAL seconds."
    sleep \$CHECK_INTERVAL
done
EOL

# Mark monitor.sh executable and start tmux
chmod +x monitor.sh
echo "→ monitor.sh is now executable."

# Launch in tmux
if command -v tmux >/dev/null 2>&1; then
    echo "→ Starting tmux session 'monitor'..."
    tmux new-session -d -s monitor './monitor.sh'
    echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] All files generated and monitor started in tmux session 'monitor'."
else
    echo "⚠️ tmux not found. Please install tmux and run './monitor.sh' manually."
fi
