#!/bin/bash
# One subprocess per song: keeps peak RSS flat and isolates failures.
cd /home/user/OldAlbum || exit 1
export MALLOC_ARENA_MAX=1
for song in 1_I_Need_You 2_Lonesome_Feeling 3_Put_It_On 4_Whats_New_Pussycat; do
    echo "=== $song $(date +%H:%M:%S)"
    /home/user/OldAlbum/.venv/bin/python -u work/restore.py "$song" >> work/logs/restore_all.log 2>&1 || echo "FAILED $song"
done
echo "ALL RESTORED $(date +%H:%M:%S)"
