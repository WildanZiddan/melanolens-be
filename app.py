import gradio as gr
from main import app as fastapi_app

with gr.Blocks(title="Melanolens AI Backend API") as demo:
    gr.Markdown("# 🔬 Melanolens AI Backend API is Live!")
    gr.Markdown("Vision Transformer (ViT) PyTorch Model Server running on 16GB RAM.")

# Mount FastAPI app onto Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
