#!/usr/bin/env bash
set -euo pipefail

KEY_PATH="${SYNC_TANK_FLEET_KEY:-$HOME/.ssh/sync_tank_fleet_ed25519}"
NODES=(
  "one@TANK_ONE_WIRED_IP"
  "two@TANK_TWO_WIRED_IP"
)

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [[ ! -f "$KEY_PATH" ]]; then
  ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "sync-tank-fleet-$(hostname)"
fi

for node in "${NODES[@]}"; do
  echo "Installing Sync Tank fleet key on $node"
  if command -v ssh-copy-id >/dev/null 2>&1; then
    ssh-copy-id -i "${KEY_PATH}.pub" \
      -o StrictHostKeyChecking=accept-new \
      "$node"
  else
    cat "${KEY_PATH}.pub" | ssh -o StrictHostKeyChecking=accept-new "$node" \
      "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
  fi
done

cat <<EOF

Fleet SSH bootstrap complete.

Use this key explicitly:
  ssh -i "$KEY_PATH" one@TANK_ONE_WIRED_IP
  ssh -i "$KEY_PATH" two@TANK_TWO_WIRED_IP

Or add it to your agent:
  ssh-add "$KEY_PATH"

Then verify:
  cd /home/zero/sync-tank
  ./scripts/fleet_agent.py --batch discover
EOF
