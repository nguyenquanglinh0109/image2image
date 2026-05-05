#!/bin/bash
SESSION_NAME="img2img"

# Nếu session đã tồn tại thì attach luôn
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attaching..."
    tmux attach-session -t $SESSION_NAME
    exit 0
fi

# Lấy thư mục chứa script bash này
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tạo session mới và chạy main.py
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd $SCRIPT_DIR && python3 main.py" C-m

# Attach vào session
tmux attach-session -t $SESSION_NAME