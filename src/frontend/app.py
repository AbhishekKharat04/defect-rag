"""Gradio frontend for the Vision-Language RAG Defect Inspector.

Provides a browser-based UI at ``http://localhost:7860`` with:

- Image upload and question input for defect analysis.
- Real-time backend API calls to ``/query`` and ``/index/dataset``.
- A visual gallery of retrieved nearest-neighbour reference images.
- Accordion controls for database indexing and dataset management.
"""

import logging
import os

import gradio as gr
import requests

from src.config import settings

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("frontend")

def run_rag(image_path, question, top_k):
    """Calls the backend API to retrieve visual context and generate grounded answer."""
    if not image_path:
        return "Please upload an image.", "unknown", "unknown", 0.0, []

    url = f"{settings.BACKEND_URL}/query"

    try:
        # Open and prepare file for post upload
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            data = {"question": question, "top_k": int(top_k)}

            logger.info(f"Sending RAG query to backend: {url}")
            response = requests.post(url, files=files, data=data)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Error from backend API: {error_detail}", "error", "error", 0.0, []

        res = response.json()

        # Prepare retrieved gallery matches
        # Each item in gallery should be a tuple (image_path/URL, caption)
        gallery_items = []
        for idx, match in enumerate(res.get("retrieved_matches", [])):
            payload = match.get("payload", {})
            path = payload.get("image_path")
            defect = payload.get("defect_label")
            severity = payload.get("severity")
            score = match.get("score")

            caption = f"Match {idx+1} | Sim: {score:.4f}\nDefect: {defect} | Severity: {severity}"
            if os.path.exists(path):
                gallery_items.append((path, caption))
            else:
                logger.warning(f"Retrieved image path does not exist locally: {path}")

        return (
            res.get("answer"),
            res.get("predicted_defect", "unknown"),
            res.get("predicted_severity", "unknown"),
            float(res.get("confidence", 0.0)),
            gallery_items
        )
    except Exception as e:
        logger.error(f"Failed to connect to backend RAG API: {e}")
        return f"Failed to connect to backend at {settings.BACKEND_URL}: {e}", "error", "error", 0.0, []

def trigger_indexing(category, synthetic, recreate):
    """Calls the backend API to index the selected dataset category."""
    url = f"{settings.BACKEND_URL}/index/dataset"
    params = {"category": category, "synthetic": bool(synthetic), "recreate": bool(recreate)}

    try:
        logger.info(f"Triggering database indexing: {url} with params {params}")
        response = requests.post(url, params=params)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Indexing failed: {error_detail}"

        res = response.json()
        col_info = res.get("collection_info", {})
        return (
            f"Success! {res.get('message')}\n\n"
            f"Collection Status: {col_info.get('status')}\n"
            f"Indexed Vectors Count: {col_info.get('vectors_count')}"
        )
    except Exception as e:
        logger.error(f"Failed to connect to backend Indexing API: {e}")
        return f"Failed to connect to backend indexing endpoint: {e}"

# Build Custom Styled Gradio Interface
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

# Custom Premium CSS Styling
css = """
body, .gradio-container {
    background: linear-gradient(135deg, #0a0f1e 0%, #121829 100%) !important;
    color: #f1f5f9 !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
}
.dashboard-panel {
    background: rgba(30, 41, 59, 0.25) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    padding: 24px !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}
.kpi-card {
    background: rgba(30, 41, 59, 0.45) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    text-align: center !important;
}
.kpi-card:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15) !important;
}
.defect-kpi {
    border-left: 4px solid #ef4444 !important;
}
.severity-kpi {
    border-left: 4px solid #f59e0b !important;
}
.confidence-kpi {
    border-left: 4px solid #10b981 !important;
}
.kpi-card label span {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
    display: block !important;
    text-align: center !important;
    margin-bottom: 6px !important;
}
.kpi-card input, .kpi-card textarea, .kpi-card .wrap, .kpi-card .container, .kpi-card .gradio-textbox, .kpi-card .gradio-number {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    text-align: center !important;
    padding: 0 !important;
}
#analyze-btn {
    background: linear-gradient(90deg, #3b82f6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
}
#analyze-btn:active {
    transform: scale(0.98) !important;
}
.gradio-container h1 {
    text-align: center !important;
    background: linear-gradient(90deg, #60a5fa 0%, #a5b4fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    margin-top: 0.5em !important;
    margin-bottom: 0.2em !important;
}
.gradio-container h3 {
    color: #e2e8f0 !important;
    margin-top: 0.5em !important;
}
#indexing-accordion {
    margin-top: 16px !important;
    background: rgba(15, 23, 42, 0.2) !important;
}
.gradio-container .gallery-item {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    background: rgba(15, 23, 42, 0.3) !important;
    overflow: hidden !important;
}
"""

with gr.Blocks(theme=theme, css=css, title="Industrial Defect RAG Assistant") as demo:
    gr.Markdown(
        """
        # 🔍 Vision-Language RAG Defect Inspector
        ### Multimodal Retrieval-Augmented Generation for Industrial Inspection and Quality Control
        """
    )

    with gr.Row():
        # --- LEFT COLUMN (Control Panel) ---
        with gr.Column(scale=4, elem_classes=["dashboard-panel"]):
            gr.Markdown("### ⚙️ Inspection Configuration")

            query_image = gr.Image(
                label="Upload Part Photo",
                type="filepath",
                sources=["upload", "clipboard"]
            )

            question_input = gr.Textbox(
                label="Verification Request",
                value="Identify the defect in this image, explain why you classified it this way, and rate the defect severity.",
                placeholder="Ask a question about the part...",
                lines=3
            )

            top_k_slider = gr.Slider(
                label="Retrieve Reference Examples (Top-K)",
                minimum=1,
                maximum=5,
                value=3,
                step=1
            )

            submit_btn = gr.Button("⚡ Analyze Part", variant="primary", elem_id="analyze-btn")

            # Indexing Panel (Accordion)
            with gr.Accordion("📦 Database Indexing Controls", open=False, elem_id="indexing-accordion"):
                gr.Markdown("Initialize or rebuild the visual search knowledge base.")
                category_dropdown = gr.Dropdown(
                    label="Dataset Category",
                    choices=["bottle", "hazelnut", "cable", "metal_nut"],
                    value="bottle"
                )
                synthetic_checkbox = gr.Checkbox(
                    label="Use Synthetic Defect Generator (Fast, No Download)",
                    value=True
                )
                recreate_checkbox = gr.Checkbox(
                    label="Recreate Collection (Delete existing entries)",
                    value=True
                )
                index_btn = gr.Button("Index Dataset Category", variant="secondary")
                index_output = gr.Textbox(label="Indexing Status", interactive=False)

        # --- RIGHT COLUMN (Inspection Results) ---
        with gr.Column(scale=6, elem_classes=["dashboard-panel"]):
            gr.Markdown("### 📋 Inspection Decision & Grounding Report")

            # Tag cards
            with gr.Row():
                defect_card = gr.Textbox(label="Predicted Defect Type", interactive=False, elem_classes=["kpi-card", "defect-kpi"])
                severity_card = gr.Textbox(label="Severity Level", interactive=False, elem_classes=["kpi-card", "severity-kpi"])
                confidence_card = gr.Number(label="Confidence Score", interactive=False, elem_classes=["kpi-card", "confidence-kpi"])

            # Grounding explanation
            answer_markdown = gr.Markdown(
                value="*Results will appear here after clicking 'Analyze Part'...*",
                line_breaks=True
            )

            gr.Markdown("### 🖼️ Retrieved Visual Database References")
            retrieved_gallery = gr.Gallery(
                label="Nearest Visual Neighbors (Qdrant Database Matches)",
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain"
            )

    # Wire actions
    submit_btn.click(
        fn=run_rag,
        inputs=[query_image, question_input, top_k_slider],
        outputs=[answer_markdown, defect_card, severity_card, confidence_card, retrieved_gallery]
    )

    index_btn.click(
        fn=trigger_indexing,
        inputs=[category_dropdown, synthetic_checkbox, recreate_checkbox],
        outputs=[index_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
