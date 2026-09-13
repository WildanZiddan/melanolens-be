# Lean, MobileNetV2-only serving image.
# The 328 MB ViT checkpoint is dead code in this app; do not ship it.
FROM python:3.11-slim

WORKDIR /app

# Build wheels before copying code (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# App source + the only model we serve (~10 MB). ViT is excluded via .dockerignore.
COPY . .

# Belt-and-braces: make sure the unused ViT never lands in the image
RUN rm -f models/ViT_B_16_Standard_70_15_15.pth

ENV PORT=8000
EXPOSE 8000

# FastAPI entrypoint (gating_service -> GatedMobileNetV2). Model load is lazy in
# a background thread on startup, so the container becomes ready before inference.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]