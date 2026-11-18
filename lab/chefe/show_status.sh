#!/usr/bin/env bash
# Opens (or reopens) the live status window on this PC's screen.
export DISPLAY=:0
U=$(id -u)
for x in /run/user/$U/gdm/Xauthority "$HOME/.Xauthority"; do [ -f "$x" ] && export XAUTHORITY="$x" && break; done
pkill -f "python3 .*lab_status.py --tui" 2>/dev/null
setsid xterm -T "Dory Lab" -geometry 110x30 -fa Monospace -fs 13 -bg black -fg white \
  -e python3 "$HOME/dory-lab/lab_status.py" --tui >/dev/null 2>&1 < /dev/null &
sleep 2
pgrep -f "python3 .*lab_status.py --tui" >/dev/null && echo "janela aberta (XAUTHORITY=$XAUTHORITY)" || echo "falhou"
