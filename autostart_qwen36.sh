#!/usr/bin/env bash
set -euo pipefail

echo "=================================================="
echo " Qwen3.6 35B Q6_K_P RunPod Bootstrap"
echo "=================================================="

MODEL_DIR="/workspace/models"
MODEL_FILE="$MODEL_DIR/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"
MODEL_REPO="HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive"
MODEL_NAME="Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf"

LLAMA_DIR="/workspace/llama.cpp"
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"

API_KEY_FILE="/workspace/qwen_api_key.txt"
START_SCRIPT="/workspace/start_qwen36_q6.sh"
LOG_FILE="/workspace/qwen36-q6-server.log"

PORT="11434"
CTX_SIZE="32768"

echo
echo "Step 1/7: installing base packages..."

apt-get update

apt-get install -y \
  git \
  cmake \
  build-essential \
  libcurl4-openssl-dev \
  pkg-config \
  python3-pip \
  pciutils \
  jq \
  tmux \
  curl \
  ca-certificates

python3 -m pip install -U "huggingface_hub[cli]" openai

echo
echo "Step 2/7: preflight check..."

echo "Checking NVIDIA GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found."
  echo "GPU driver/tools are unavailable."
  exit 1
fi

nvidia-smi || {
  echo "ERROR: NVIDIA GPU is not available."
  exit 1
}

echo "Checking CUDA compiler..."
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found."
  echo "You probably selected a runtime image instead of a devel image."
  echo
  echo "Use this RunPod container image:"
  echo "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  exit 1
fi

echo "Checking required commands..."
for cmd in git cmake g++ python3 pip jq tmux huggingface-cli curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd"
    exit 1
  fi
done

echo "Preflight OK."

echo
echo "Step 3/7: checking API key..."

if [ ! -f "$API_KEY_FILE" ]; then
  echo "API key not found. Creating new API key..."
  python3 - <<'PY'
import secrets
key = "rp_" + secrets.token_urlsafe(32)
open("/workspace/qwen_api_key.txt", "w").write(key + "\n")
print("New API key created in /workspace/qwen_api_key.txt")
PY
else
  echo "API key exists: $API_KEY_FILE"
fi

echo
echo "Step 4/7: checking model file..."

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
  SIZE_BYTES="$(stat -c%s "$MODEL_FILE")"
  MIN_BYTES=$((25 * 1024 * 1024 * 1024))

  if [ "$SIZE_BYTES" -lt "$MIN_BYTES" ]; then
    echo "Model file looks incomplete. Removing it..."
    rm -f "$MODEL_FILE"
  else
    echo "Model file exists and looks OK:"
    ls -lh "$MODEL_FILE"
  fi
fi

if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading model:"
  echo "$MODEL_REPO / $MODEL_NAME"

  huggingface-cli download "$MODEL_REPO" \
    "$MODEL_NAME" \
    --local-dir "$MODEL_DIR"
fi

echo
echo "Step 5/7: checking llama.cpp..."

if [ ! -d "$LLAMA_DIR/.git" ]; then
  echo "llama.cpp not found. Cloning..."
  rm -rf "$LLAMA_DIR"
  cd /workspace
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
else
  echo "llama.cpp exists."
fi

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server not found. Building llama.cpp with CUDA..."

  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON

  cmake --build "$LLAMA_DIR/build" \
    --config Release \
    -j"$(nproc)" \
    --target llama-server llama-cli
else
  echo "llama-server exists:"
  "$LLAMA_SERVER" --version || true
fi

echo
echo "Step 6/7: creating start script..."

cat > "$START_SCRIPT" <<EOS
#!/usr/bin/env bash
set -euo pipefail

MODEL="$MODEL_FILE"
API_KEY="\$(cat "$API_KEY_FILE")"

echo "Starting Qwen3.6 Q6_K_P..."
echo "Model: \$MODEL"
echo "Port: $PORT"
echo "Context: $CTX_SIZE"
echo "API key file: $API_KEY_FILE"

/workspace/llama.cpp/build/bin/llama-server \\
  -m "\$MODEL" \\
  --host 0.0.0.0 \\
  --port $PORT \\
  --api-key "\$API_KEY" \\
  --jinja \\
  -c $CTX_SIZE \\
  -ngl 99 \\
  --flash-attn auto \\
  --parallel 1
EOS

chmod +x "$START_SCRIPT"

echo
echo "Step 7/7: starting server..."
echo "Log file: $LOG_FILE"
echo
echo "When server is ready, open RunPod HTTP port 11434."
echo "API key is stored inside the pod:"
echo "$API_KEY_FILE"
echo

"$START_SCRIPT" 2>&1 | tee "$LOG_FILE"
