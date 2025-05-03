#!/usr/bin/env bash
set -euo pipefail

### ----------------------------------------------------------------------------
### CONFIGURATION: tweak these if you like to non-interactive defaults
### ----------------------------------------------------------------------------
DEFAULT_NETUID=56
DEFAULT_SUBNETWORK="finney"
DEFAULT_STAKE="1000"
SCREEN_NAME="miner"
### ----------------------------------------------------------------------------

function initial_install() {
  echo "🛠  Beginning initial install…"

  # 1) Prompt for keys + tokens
  read -p "Enter your Coldkey MNEMONIC (words separated by spaces): " -r COLDKEY_MNEMONIC
  read -p "Enter your Hotkey  MNEMONIC (words separated by spaces): " -r HOTKEY_MNEMONIC
  read -p "HuggingFace username (for model uploads): " HF_USER
  read -p "HuggingFace token: "            HF_TOKEN
  read -p "Weights & Biases token: "        WANDB_TOKEN
  read -p "Wallet name (default: default): "               WALLET_NAME
  WALLET_NAME=${WALLET_NAME:-default}
  read -p "Hotkey name  (default: default): "               HOTKEY_NAME
  HOTKEY_NAME=${HOTKEY_NAME:-default}

  export BT_COLDKEY_MNEMONIC="$COLDKEY_MNEMONIC"
  export BT_HOTKEY_MNEMONIC="$HOTKEY_MNEMONIC"

  ### 2) Install bittensor-cli & register
  echo "⬇️  Downloading bittensor-cli installer…"
  wget -q https://raw.githubusercontent.com/rennzone/Auto-Install-Bittensor-Script/refs/heads/main/bittensor-cli.sh
  bash bittensor-cli.sh

  echo "🔗 Registering on subnet ${DEFAULT_NETUID}…"
  btcli subnet register \
    --netuid ${DEFAULT_NETUID} \
    --wallet.name "${WALLET_NAME}" \
    --wallet.hotkey "${HOTKEY_NAME}"

  ### 3) Clone G.O.D & cooking
  echo "⬇️  Cloning G.O.D…"
  git clone https://github.com/rayonlabs/G.O.D.git

  echo "⬇️  Downloading cooking.sh for chain ${DEFAULT_NETUID}…"
  wget -q https://raw.githubusercontent.com/luminousify/56/refs/heads/main/cooking.sh
  bash cooking.sh

  ### 4) Bootstrap G.O.D
  echo "🚀 Bootstrapping G.O.D…"
  cd G.O.D
  sudo -E ./bootstrap.sh

  ### 5) Activate env + install tasks
  echo "🔄 Sourcing shell + activating virtualenv…"
  source "$HOME/.bashrc"
  source "$HOME/.venv/bin/activate"

  echo "✅ Installing tasks…"
  task install

  ### 6) Auto-generate miner config via here-doc
  echo "🤖 Generating miner .env…"
  python3 -m core.create_config --miner <<EOF
${DEFAULT_SUBNETWORK}

${WALLET_NAME}
${HOTKEY_NAME}
${WANDB_TOKEN}
${HF_TOKEN}
${HF_USER}
${DEFAULT_STAKE}
EOF

  echo "💡 Miner config written to .prod.env (netuid=${DEFAULT_NETUID}, subnetwork=${DEFAULT_SUBNETWORK})"

  ### 7) Ready to reboot
  echo -e "\n⚡ All set—rebooting now to pick up drivers & modules…"
  echo "When VPS is back, run:   ./deploy.sh post-reboot"
  sudo reboot
}

function post_reboot() {
  echo "🔍 Post-reboot steps…"

  echo -e "\n1) Verifying GPU driver:"
  nvidia-smi || { echo "❌ nvidia-smi failed—check your driver install"; exit 1; }

  echo -e "\n2) Publishing your external IP + starting miner in screen..."
  cd G.O.D

  # Grab your IP
  EXTERNAL_IP=$(curl -s https://ifconfig.me)

  screen -dmS ${SCREEN_NAME} bash -c "
    fiber-post-ip \
      --netuid ${DEFAULT_NETUID} \
      --subtensor.network ${DEFAULT_SUBNETWORK} \
      --external_port 7999 \
      --wallet.name ${WALLET_NAME} \
      --wallet.hotkey ${HOTKEY_NAME} \
      --external_ip ${EXTERNAL_IP}

    task miner
  "

  echo -e "\n✅ Miner is running in detached screen session '${SCREEN_NAME}'."
  echo "   Attach with: screen -r ${SCREEN_NAME}"
}

### === MAIN ===
if [[ "${1:-}" == "post-reboot" ]]; then
  post_reboot
else
  initial_install
fi
