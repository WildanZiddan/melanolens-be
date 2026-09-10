import gradio as gr
from main import app as fastapi_app
import os

with gr.Blocks(title="Melanolens AI Backend API") as demo:
    gr.Markdown("# 🔬 Melanolens AI Backend API is Live!")
    gr.Markdown("Vision Transformer (ViT) PyTorch Model Server running on 16GB RAM.")

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
