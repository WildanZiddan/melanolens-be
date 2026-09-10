import gradio as gr
from main import app as fastapi_app

# Mount FastAPI app onto Gradio (Full FastAPI REST API Support with 16GB RAM)
app = gr.mount_gradio_app(fastapi_app, gr.Blocks(title=" Melanolens AI Backend API\), path=\/\)
