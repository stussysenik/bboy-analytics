#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Extract sequence clips from je265bdPIEU (BC One 2018)
# Source: 25fps, 1920x1080, The Wolfer vs Luigi
#
# Usage: bash poc/remote/extract-clips-je265.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VIDEO="${SCRIPT_DIR}/data/brace/videos/je265bdPIEU.mp4"
OUTDIR="${SCRIPT_DIR}/data/brace/videos"

if [ ! -f "${VIDEO}" ]; then
    echo "ERROR: ${VIDEO} not found. Download first with yt-dlp."
    exit 1
fi

FPS=25

echo "╔══════════════════════════════════════════╗"
echo "║  Extract je265bdPIEU sequence clips       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Seq 2: The Wolfer — POWERMOVE (frames 4821-5134, 313 frames, ~12.5s)
echo "▸ Seq 2: The Wolfer — powermove (4821-5134)..."
ffmpeg -y -i "${VIDEO}" \
    -vf "select='between(n\,4821\,5134)',setpts=PTS-STARTPTS" \
    -an -c:v libx264 -crf 18 \
    "${OUTDIR}/je265bdPIEU_seq2.mp4" 2>/dev/null
echo "  Saved: je265bdPIEU_seq2.mp4 (video only)"

# Seq 2 with audio (for musicality analysis)
START_SEC=$(python3 -c "print(4821/25)")
DURATION_SEC=$(python3 -c "print((5134-4821+1)/25)")
ffmpeg -y -i "${VIDEO}" \
    -ss "${START_SEC}" -t "${DURATION_SEC}" \
    -c:v libx264 -crf 18 -c:a aac \
    "${OUTDIR}/je265bdPIEU_seq2_audio.mp4" 2>/dev/null
echo "  Saved: je265bdPIEU_seq2_audio.mp4 (with audio)"

# Seq 0: The Wolfer — toprock + footwork (frames 2968-3780, 812 frames, ~32.5s)
echo ""
echo "▸ Seq 0: The Wolfer — toprock+footwork (2968-3780)..."
ffmpeg -y -i "${VIDEO}" \
    -vf "select='between(n\,2968\,3780)',setpts=PTS-STARTPTS" \
    -an -c:v libx264 -crf 18 \
    "${OUTDIR}/je265bdPIEU_seq0.mp4" 2>/dev/null
echo "  Saved: je265bdPIEU_seq0.mp4 (video only)"

# Seq 0 with audio
START_SEC=$(python3 -c "print(2968/25)")
DURATION_SEC=$(python3 -c "print((3780-2968+1)/25)")
ffmpeg -y -i "${VIDEO}" \
    -ss "${START_SEC}" -t "${DURATION_SEC}" \
    -c:v libx264 -crf 18 -c:a aac \
    "${OUTDIR}/je265bdPIEU_seq0_audio.mp4" 2>/dev/null
echo "  Saved: je265bdPIEU_seq0_audio.mp4 (with audio)"

# Seq 1: Luigi — toprock + footwork (frames 3880-4723, 843 frames, ~33.7s)
echo ""
echo "▸ Seq 1: Luigi — toprock+footwork (3880-4723)..."
ffmpeg -y -i "${VIDEO}" \
    -vf "select='between(n\,3880\,4723)',setpts=PTS-STARTPTS" \
    -an -c:v libx264 -crf 18 \
    "${OUTDIR}/je265bdPIEU_seq1.mp4" 2>/dev/null
echo "  Saved: je265bdPIEU_seq1.mp4 (video only)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Done! Clips in data/brace/videos/        ║"
echo "╚══════════════════════════════════════════╝"

# Verify
echo ""
echo "▸ Verification:"
for f in je265bdPIEU_seq0.mp4 je265bdPIEU_seq1.mp4 je265bdPIEU_seq2.mp4 je265bdPIEU_seq2_audio.mp4 je265bdPIEU_seq0_audio.mp4; do
    if [ -f "${OUTDIR}/${f}" ]; then
        FRAMES=$(ffprobe -v quiet -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 "${OUTDIR}/${f}" 2>/dev/null || echo "?")
        SIZE=$(du -h "${OUTDIR}/${f}" | cut -f1)
        echo "  ${f}: ${FRAMES} frames, ${SIZE}"
    fi
done
