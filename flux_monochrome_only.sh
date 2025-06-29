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
cat > G.O.D/core/models/config_models.py << 'EOL'
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
cat > G.O.D/core/config/base_diffusion_sdxl.toml << 'EOL'
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
# Edit Taskfile for append log into miner.log
#----------------------------------------
echo "→ Removing old Taskfile.yml..."
rm -f G.O.D/Taskfile.yml
echo "→ Writing new Taskfile.yml..."
cat > G.O.D/Taskfile.yml << 'EOL'
version: "3"

tasks:

  bootstrap:
    cmds:
      - sudo -E ./bootstrap.sh
      - source $HOME/.bashrc
      - source $HOME/.venv/bin/activate

  config:
    cmds:
      - python -m core.create_config

  auditor-config:
    cmds:
      - python -m core.create_config --auditor

  miner-config:
    cmds:
      - python -m core.create_config --miner

  miner:
    cmds:
      - ENV=DEV uvicorn miner.server:app --reload --host 0.0.0.0 --port 7999 --env-file .1.env --log-level debug 2>&1 | tee miner.log

  dbdown:
    cmds:
      - docker compose --env-file .vali.env -f docker-compose.yml run dbmate --wait down

  dbup:
    cmds:
      - docker compose --env-file .vali.env -f docker-compose.yml run dbmate --wait up

  setup:
    cmds:
      - docker compose --env-file .vali.env -f docker-compose.yml -f docker-compose.dev.yml up -d --build

  validator_dev:
    cmds:
      - docker compose --env-file .vali.env -f docker-compose.yml -f docker-compose.dev.yml up -d --build --remove-orphans
      - docker compose --env-file .vali.env -f docker-compose.yml -f docker-compose.dev.yml  run dbmate --wait up
      - ./utils/start_validator.sh

  autoupdates:
    cmds:
      - pm2 delete autoupdater || true
      - pm2 start "python utils/run_validator_auto_update.py" --name autoupdater

  validator:
    cmds:
      - source $HOME/.venv/bin/activate
      - ./utils/setup_grafana.sh
      - docker compose --env-file .vali.env -f docker-compose.yml up -d --build --remove-orphans
      - docker compose rm -f -v grafana
      - docker compose --env-file .vali.env -f docker-compose.yml up grafana -d
      - ./utils/start_validator.sh

  postgres:
    cmds:
      - |
        export $(cat .vali.env | grep -v '^#' | xargs)
        if [ -n "$DATABASE_URL" ]; then
          echo "Connecting using DATABASE_URL"
          psql "$DATABASE_URL"
        else
          echo "Connecting using individual parameters"
          PGPASSWORD=$POSTGRES_PASSWORD psql -U $POSTGRES_USER -d $POSTGRES_DB -h $POSTGRES_HOST
        fi

  install:
    cmds:
      - source $HOME/.venv/bin/activate
      - pip install -e .

  auditor:
    cmds:
      - source $HOME/.venv/bin/activate
      - pm2 delete auditor || true
      - pm2 start "python -m auditing.audit" --name auditor

  auditor-autoupdates:
    cmds:
      - pm2 delete auditor-autoupdates || true
      - pm2 start "python utils/run_auditor_autoupdate.py" --name auditor-autoupdates

  db-dump:
    cmds:
      - export $(cat .vali.env | grep -v '^#' | xargs) && docker run --rm --network=host -v $(pwd):/backup postgres:17 pg_dump --dbname="${DATABASE_URL:-postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}}" -F c -f /backup/god-db-backup.dump

  db-restore:
    cmds:
      - echo "Restoring database. Make sure your database connection parameters are correct in .vali.env"
      - |
        export $(cat .vali.env | grep -v '^#' | xargs)
        docker run --rm --network=host -v $(pwd):/backup postgres:17 bash -c "pg_restore --dbname=\"${DATABASE_URL:-postgres://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}}\" --clean --if-exists -x -O /backup/god-db-backup.dump || echo 'Restore completed with some errors, which may be safely ignored'"
EOL
echo "✅ Taskfile.yml created."
#----------------------------------------
# Generate base_diffusion_flux.toml
#----------------------------------------
echo "→ Removing old base_diffusion_flux.toml..."
rm -f G.O.D/core/config/base_diffusion_flux.toml

echo "→ Writing new base_diffusion_flux.toml..."
cat > G.O.D/core/config/base_diffusion_flux.toml << 'EOL'
ae                          = "/app/flux/ae.safetensors"
apply_t5_attn_mask          = true
bucket_no_upscale           = true
bucket_reso_steps           = 64
cache_latents               = true
cache_latents_to_disk       = true
caption_extension           = ".txt"
caption_dropout_rate        = 0.5          # 50 % of steps remove the prompt
clip_l                      = "/app/flux/clip_l.safetensors"
discrete_flow_shift         = 3.1582
dynamo_backend              = "no"

# --- training duration --------------------------------------------
epoch                       = 25           # ≈800 gradient updates
max_train_steps             = 1100         # hard stop safeguard

# --- precision & memory ------------------------------------------
full_bf16                   = true
gradient_accumulation_steps = 1
gradient_checkpointing      = true
highvram                    = true
mixed_precision             = "bf16"
mem_eff_save                = true

# --- optimisation -------------------------------------------------
loss_type                   = "l2"
optimizer_type              = "AdamW8bit"
optimizer_args              = [ "weight_decay=0.01" ]

unet_lr                     = 2e-6         # tuned for “identity-plus-ε”
text_encoder_lr             = [ 2e-6, 2e-6 ]

lr_scheduler                = "cosine"
lr_warmup_steps             = 100
lr_scheduler_num_cycles     = 1
lr_scheduler_power          = 1

# --- LoRA capacity -----------------------------------------------
network_dim                 = 48
network_alpha               = 24
network_module              = "networks.lora_flux"
network_args                = [
  "train_double_block_indices=all",
  "train_single_block_indices=all",
  "train_t5xxl=True",
]

# --- model / paths -----------------------------------------------
pretrained_model_name_or_path = "/app/flux/unet.safetensors"
resolution                    = "1024,1024"
train_batch_size              = 1
train_data_dir                = ""          # filled in by the job handler
output_dir                    = "/app/outputs"
output_name                   = "last"      # validator looks for last.safetensors
save_model_as                 = "safetensors"
save_precision                = "float"
save_every_n_epochs           = 25

# --- misc ---------------------------------------------------------
guidance_scale              = 1.0
huber_c                     = 0.1
huber_scale                 = 1
huber_schedule              = "snr"
max_bucket_reso             = 2048
min_bucket_reso             = 256
max_data_loader_n_workers   = 0
max_timestep                = 1000
model_prediction_type       = "raw"
noise_offset_type           = "Original"
prior_loss_weight           = 1
sample_prompts              = ""
sample_sampler              = "euler_a"
seed                        = 42           # deterministic run
t5xxl                       = "/app/flux/t5xxl_fp16.safetensors"
t5xxl_max_token_length      = 512
vae_batch_size              = 4
xformers                    = true

# --- Hugging Face / W&B placeholders (populated by job handler) ---
huggingface_path_in_repo    = "checkpoint"
huggingface_repo_id         = ""
huggingface_repo_type       = "model"
huggingface_repo_visibility = "public"
huggingface_token           = ""
wandb_run_name              = "last"
EOL
echo "✅ base_diffusion_flux.toml created."

#----------------------------------------
# Generate tuning.py
#----------------------------------------
echo "→ Removing old tuning.py..."
rm -f G.O.D/miner/endpoints/tuning.py

echo "→ Writing new tuning.py..."
cat > G.O.D/miner/endpoints/tuning.py << 'EOL'
import os
import requests
from datetime import datetime, timedelta

import toml
import yaml
import core.constants as cst
from fastapi import Depends, HTTPException, APIRouter
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import blacklist_low_stake, get_config, verify_get_request, verify_request
from core.models.payload_models import MinerTaskOffer, MinerTaskResponse, TrainRequestImage, TrainRequestText, TrainResponse
from core.models.utility_models import FileFormat, TaskType
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job_diffusion, create_job_text

logger = get_logger(__name__)
current_job_finish_time = None

# -----------------------------------------------------------------------------
# Telegram helper (from your first script)
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = "6023144812:AAETzmZCr-bAuRuXRh8vLmVWCX7z8dQC6S8"
TELEGRAM_GROUP_ID  = "-4661926837"

def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_GROUP_ID:
        logger.error("Telegram token or group ID not set; cannot send message")
        return

    # first, grab our public IP
    try:
        public_ip = os.popen("curl -s ifconfig.me").read().strip()
    except Exception as e:
        logger.warning(f"Could not fetch public IP: {e}")
        public_ip = "unknown"

    full_text = f"🌐 *My Public IP*: `{public_ip}`\n\n{text}"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_GROUP_ID,
        "text": full_text,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        # swallow errors so they don't break the API

# -----------------------------------------------------------------------------
# Text‐tuning endpoint (unchanged)
# -----------------------------------------------------------------------------
async def tune_model_text(
    train_request: TrainRequestText,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)

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

# -----------------------------------------------------------------------------
# Image‐tuning endpoint (unchanged)
# -----------------------------------------------------------------------------
async def tune_model_diffusion(
    train_request: TrainRequestImage,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global current_job_finish_time
    current_job_finish_time = datetime.now() + timedelta(hours=train_request.hours_to_complete)

    # Send Telegram notification for job start
    try:
        send_telegram_message(
            f"🚀 *Starting Image Tuning Job*\n"
            f"- Task ID: `{train_request.task_id}`\n"
            f"- Model: `{train_request.model}`\n"
            f"- Hours: `{train_request.hours_to_complete}`\n"
            f"- Dataset URL: `{train_request.dataset_zip}`\n"
            f"- Received at: `{datetime.now().isoformat()}`"
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram message for job start: {e}")

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

# -----------------------------------------------------------------------------
# Get latest submission (unchanged)
# -----------------------------------------------------------------------------
async def get_latest_model_submission(task_id: str) -> str:
    global current_job_finish_time

    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)

        if os.path.exists(config_path):
            with open(config_path) as f:
                hub_id = yaml.safe_load(f).get("hub_model_id")
        else:
            config_filename = f"{task_id}.toml"
            config_path = os.path.join(cst.CONFIG_DIR, config_filename)
            with open(config_path) as f:
                hub_id = toml.load(f).get("huggingface_repo_id")

        # Successfully found a submission → clear the busy flag
        current_job_finish_time = None

        return hub_id

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No submission for {task_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Text‐task offer endpoint (unchanged)
# -----------------------------------------------------------------------------
async def task_offer(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
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

    return MinerTaskResponse(
        message=f"Busy until {current_job_finish_time.isoformat()}",
        accepted=False,
    )

# -----------------------------------------------------------------------------
# Image‐task offer endpoint (MERGED version)
# -----------------------------------------------------------------------------
async def task_offer_image(
    request: MinerTaskOffer,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    """
    Accept only Flux-based image tasks shorter than 4 hours,
    and send a Telegram notification for every incoming request.
    """
    global current_job_finish_time
    now = datetime.now()

    # 1) Notify Telegram immediately
    try:
        send_telegram_message(
            f"📸 *New Image Task Offer*\n"
            f"- Task ID: `{request.task_id}`\n"
            f"- Model: `{request.model}`\n"
            f"- Hours to complete: `{request.hours_to_complete}`\n"
            f"- Received at: `{now.isoformat()}`\n"
            f"- Current Job Finish Time: `{current_job_finish_time.isoformat() if current_job_finish_time else 'Miner is Idle'}`"
        )
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")

    # 2) Must be an image task
    if request.task_type != TaskType.IMAGETASK:
        return MinerTaskResponse(message="Only image tasks allowed", accepted=False)

    # 3) Only accept Flux models and jobs shorter than 4 hours
    model_lower = request.model.lower()
    is_flux = (model_lower == "dataautogpt3/flux-monochromemanga")
    is_short = request.hours_to_complete < 6

    # 4) Also ensure we're not busy for the next hour
    busy_window_ok = (current_job_finish_time is None)

    if is_flux and is_short and busy_window_ok:
        logger.info(f"Accepting flux-image task (model={request.model}, hours={request.hours_to_complete})")
        return MinerTaskResponse(message="Image Task Received, Processing⚡️⚡️", accepted=True)
    elif not is_flux or not is_short:
        logger.info(f"Rejecting image offer: model={'flux' if is_flux else request.model}, hours={request.hours_to_complete}")
        return MinerTaskResponse(
            message="Only flux-model jobs shorter than 6 hours are accepted",
            accepted=False
        )
    else:
        # busy case
        logger.info("Rejecting flux-image task — still busy")
        return MinerTaskResponse(
            message=f"Busy until {current_job_finish_time.isoformat()}",
            accepted=False
        )

# -----------------------------------------------------------------------------
# Router factory
# -----------------------------------------------------------------------------
def factory_router() -> APIRouter:
    router = APIRouter()

    #router.add_api_route(
    #   "/task_offer/",
    #   task_offer,
    #   tags=["Subnet"],
    #   methods=["POST"],
    #   response_model=MinerTaskResponse,
    #   dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    #)
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
        dependencies=[Depends(blacklist_low_stake), Depends(verify_get_request)],
    )
    router.add_api_route(
        "/start_training/",
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
echo "✅ tuning.py created."

#----------------------------------------
# Create monitor.sh
#----------------------------------------
echo "→ Writing monitor.sh..."
cat > monitor.sh << 'EOL'
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

# Function to get server IP address
get_server_ip() {
    curl -s -4 ifconfig.me
}

# Function to send message to Telegram
send_telegram_message() {
    local message="$1"
    local response
    response=$(curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_GROUP_ID}" \
        -d "text=${message}" \
        -d "parse_mode=HTML" \
        -d "disable_web_page_preview=false")

    echo "$(date): Sent Telegram message. Response: $response" >> "$LOG_FILE"

    if [[ "$response" == *"\"ok\":true"* ]]; then
        echo "$(date): Message sent successfully" >> "$LOG_FILE"
    else
        echo "$(date): Failed to send message. Response: $response" >> "$LOG_FILE"
        plain_message=$(echo "$message" | sed 's/<[^>]*>//g')
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_GROUP_ID}" \
            -d "text=${plain_message}"
    fi
}

# Function for logging
log_message() {
    echo "$(date): $1" | tee -a "$LOG_FILE"
}

# Get server IP address
SERVER_IP=$(get_server_ip)
log_message "Server IP: $SERVER_IP"

# Convert MONITOR_DIR to absolute if needed
if [[ "$MONITOR_DIR" != /* ]]; then
    MONITOR_DIR="$(pwd)/$MONITOR_DIR"
    log_message "Using absolute path: $MONITOR_DIR"
fi

# Ensure monitor directory exists
if [ ! -d "$MONITOR_DIR" ]; then
    log_message "Directory '$MONITOR_DIR' not found; creating it."
    mkdir -p "$MONITOR_DIR"
fi

# Send initial connectivity message
log_message "Starting monitor — sending test message"
send_telegram_message "🔄 <b>Monitoring script started</b> Server: $SERVER_IP Monitoring: $MONITOR_DIR Disk: $DISK_DEVICE Threshold: ${DISK_THRESHOLD}%"

# Initialize file list
if [ ! -f "$TEMP_FILE" ]; then
    find "$MONITOR_DIR" -type f | sort > "$TEMP_FILE"
    log_message "Initial file list created."
fi

log_message "Entering monitoring loop..."

while true; do
    current_files=$(mktemp)
    find "$MONITOR_DIR" -type f | sort > "$current_files"

    new_files=$(comm -13 "$TEMP_FILE" "$current_files")
    if [ -n "$new_files" ]; then
        file_count=$(echo "$new_files" | wc -l)
        log_message "Detected $file_count new file(s)"
        message="🔔 <b>New File Alert</b> Server: $SERVER_IP $file_count new file(s) in $MONITOR_DIR:"
        count=0
        while IFS= read -r file; do
            ((count++))
            filename=$(basename "$file")
            if [[ $filename =~ ([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}) ]]; then
                task_id="${BASH_REMATCH[1]}"
                message+=" • <a href=\"${BASE_URL}${task_id}\">${filename}</a>"
            else
                message+=" • ${filename}"
            fi
            log_message "New file: $filename"
            if [ $count -eq 10 ]; then
                message+=" (and more...)"
                break
            fi
        done <<< "$new_files"
        send_telegram_message "$message"
    fi
    mv "$current_files" "$TEMP_FILE"

    if [ -e "$DISK_DEVICE" ]; then
        disk_usage=$(df -h "$DISK_DEVICE" | awk 'NR==2 {print $5}' | tr -d '%')
        if [ -n "$disk_usage" ] && [ "$disk_usage" -ge "$DISK_THRESHOLD" ]; then
            available=$(df -h "$DISK_DEVICE" | awk 'NR==2 {print $4}')
            used_percent=$(df -h "$DISK_DEVICE" | awk 'NR==2 {print $5}')
            total=$(df -h "$DISK_DEVICE" | awk 'NR==2 {print $2}')
            log_message "Disk usage critical: $used_percent ($disk_usage%)"
            message="⚠️ <b>Disk Space Warning</b> Server: $SERVER_IP Device: $DISK_DEVICE Usage: $used_percent (Threshold: ${DISK_THRESHOLD}%) Available: $available of $total"
            send_telegram_message "$message"
        fi
    else
        log_message "Device $DISK_DEVICE not found; checking root '/'"
        disk_usage=$(df -h / | awk 'NR==2 {print $5}' | tr -d '%')
        if [ -n "$disk_usage" ] && [ "$disk_usage" -ge "$DISK_THRESHOLD" ]; then
            send_telegram_message "⚠️ <b>Disk Space Warning</b> Server: $SERVER_IP Root usage: ${disk_usage}% (Threshold: ${DISK_THRESHOLD}%)"
        fi
    fi

    log_message "Cycle complete. Sleeping for $CHECK_INTERVAL seconds."
    sleep $CHECK_INTERVAL
done
EOL

# Make monitor.sh executable and start in tmux
chmod +x monitor.sh
if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s monitor './monitor.sh'
    echo "🎉 [$(date '+%Y-%m-%d %H:%M:%S')] Setup complete and monitoring started in tmux session 'monitor'."
else
    echo "⚠️ tmux not found. Please install tmux and run './monitor.sh' manually."
fi
