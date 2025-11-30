cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || true
cat /sys/fs/cgroup/cpuset/cpuset.mems 2>/dev/null || true

# expand top-level cpuset to include cpus 0-223 (example)
# WARNING: ensure this is safe in your environment before writing
sudo bash -c 'echo "0-223" > /sys/fs/cgroup/cpuset/cpuset.cpus'
# cpuset requires mems to be set to a valid node; set to existing node 0 if that's valid
sudo bash -c 'echo "0" > /sys/fs/cgroup/cpuset/cpuset.mems'

# if your process is in a child cpuset, write the same range to that child cpuset.cpus
# determine child path from /proc/self/cgroup and write to that file (example)
# e.g. if /proc/self/cgroup shows cpuset:/myjob then:
sudo bash -c 'echo "0-223" > /sys/fs/cgroup/myjob/cpuset.cpus'

echo "--- /proc/self/cgroup ---"; cat /proc/self/cgroup || true
echo "--- mounts (cgroup) ---"; mount | grep cgroup || true
echo "--- cgroup v2 controllers ---"; cat /sys/fs/cgroup/cgroup.controllers 2>/dev/null || true
echo "--- list /sys/fs/cgroup ---"; ls -la /sys/fs/cgroup | sed -n "1,200p"

#!/bin/bash
set -euo pipefail

TARGET_CPUS="0-223"
TARGET_MEMS="0"
CHILD="mycpuset"

echo "Detecting cgroup setup..."
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  echo "cgroup v2 detected."
  controllers=$(cat /sys/fs/cgroup/cgroup.controllers)
  echo "Available controllers: $controllers"
  if ! grep -q '\bcpuset\b' <(echo "$controllers"); then
    echo "ERROR: cpuset controller not available on this kernel. Abort."
    exit 1
  fi

  echo "Enabling cpuset controller in root cgroup..."
  # enable cpuset for children
  sudo bash -c "echo +cpuset > /sys/fs/cgroup/cgroup.subtree_control"

  # create child and set cpus/mems
  if [ ! -d "/sys/fs/cgroup/${CHILD}" ]; then
    sudo mkdir -p "/sys/fs/cgroup/${CHILD}"
    sudo chown root:root "/sys/fs/cgroup/${CHILD}"
  fi

  echo "Writing cpus/mems to /sys/fs/cgroup/${CHILD}"
  sudo bash -c "echo '${TARGET_CPUS}' > /sys/fs/cgroup/${CHILD}/cpuset.cpus"
  sudo bash -c "echo '${TARGET_MEMS}' > /sys/fs/cgroup/${CHILD}/cpuset.mems"

  echo "Move a process into the new cgroup to inherit it (example: move current shell):"
  echo "  sudo bash -c 'echo \$\$ > /sys/fs/cgroup/${CHILD}/cgroup.procs'"
  echo "Or run your training inside the cgroup by writing the pid of the launched process."

else
  echo "cgroup v1 or cpuset not mounted. Mounting cpuset at /sys/fs/cgroup/cpuset..."
  if [ ! -d /sys/fs/cgroup/cpuset ]; then
    sudo mkdir -p /sys/fs/cgroup/cpuset
  fi
  sudo mount -t cgroup -o cpuset cpuset /sys/fs/cgroup/cpuset || true

  echo "Writing cpuset.cpus and cpuset.mems (root cpuset)"
  sudo bash -c "echo '${TARGET_CPUS}' > /sys/fs/cgroup/cpuset/cpuset.cpus"
  sudo bash -c "echo '${TARGET_MEMS}' > /sys/fs/cgroup/cpuset/cpuset.mems"
fi

echo "Done. Verify with:"
echo "  cat /proc/self/status | awk '/Cpus_allowed_list/ {print \$2}'"
echo "  python - <<'PY'\nimport os\nprint('affinity:', sorted(list(os.sched_getaffinity(0))))\nPY"