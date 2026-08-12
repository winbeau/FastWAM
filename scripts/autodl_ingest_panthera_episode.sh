#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/autodl_ingest_panthera_episode.sh EPISODE_ID HF_REVISION [smoke|episodes]

Required environment:
  PANTHERA_WAM_ROOT   Restored Panthera-WAM source root containing tools/lerobot-v3

Optional environment:
  FASTWAM_DATA_ROOT   Persistent data root (default: /root/autodl-tmp/fastwam-lerobot)
  HF_REPO_ID          Dataset repo (default: winbeau/fastwam-lerobot)
  UV_CACHE_DIR        uv cache (default: /root/autodl-tmp/uv-cache)
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

episode_id=$1
revision=$2
kind=${3:-episodes}
case "$kind" in
    episodes|smoke) ;;
    *) echo "kind must be episodes or smoke" >&2; exit 2 ;;
esac

root="${FASTWAM_DATA_ROOT:-/root/autodl-tmp/fastwam-lerobot}"
repo_id="${HF_REPO_ID:-winbeau/fastwam-lerobot}"
panthera_root="${PANTHERA_WAM_ROOT:?PANTHERA_WAM_ROOT is required}"
producer="$panthera_root/tools/lerobot-v3"
staging="$root/staging/$episode_id"
dataset="$root/lerobot-v3/$episode_id"

for path in "$producer/pyproject.toml" "$producer/uv.lock"; do
    [[ -f "$path" ]] || { echo "missing producer file: $path" >&2; exit 1; }
done
[[ ! -e "$staging" ]] || { echo "staging output already exists: $staging" >&2; exit 1; }
[[ ! -e "$dataset" ]] || { echo "dataset output already exists: $dataset" >&2; exit 1; }

export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
mkdir -p "$UV_CACHE_DIR" "$root/downloads" "$root/staging" "$root/lerobot-v3"

uv run --frozen python scripts/hf_fetch_panthera_episode.py \
    "$episode_id" \
    --revision "$revision" \
    --kind "$kind" \
    --repo-id "$repo_id" \
    --download-dir "$root/downloads" \
    --output-root "$root/staging"

(
    cd "$producer"
    uv sync --frozen
    uv run --frozen panthera-pack-lerobot-v3 \
        --staging "$staging" \
        --output "$dataset" \
        --repo-id "$repo_id" \
        --vcodec h264
)

uv run --frozen python scripts/validate_panthera_dataset.py "$dataset"
printf 'Panthera episode ready: %s\n' "$dataset"
