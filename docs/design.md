# Design Decisions & Architectural Choices

This document explains the technical choices and tradeoffs made during the design and implementation of the Vision-Language RAG Assistant for industrial defect detection.

---

## 1. Vector Database: Qdrant
**Chosen Tool**: Qdrant (Self-hosted via Docker)
**Alternatives Considered**: Chroma, Milvus, Pinecone, pgvector

### Why Qdrant?
- **Performance & Efficiency**: Written in Rust, Qdrant is highly performant and memory-efficient. It scales exceptionally well from local development to production.
- **Rich Filtering capabilities**: Qdrant allows combining vector search with complex payload filtering (e.g., matching image defect categories or severity levels) without sacrificing retrieval speed.
- **Developer Experience**: Has an excellent, idiomatic Python SDK (`qdrant-client`), high-quality documentation, and a built-in web UI for inspecting vectors and collections.
- **Cost**: Open-source and self-hostable, eliminating cloud cost concerns on a free tier or local environment, unlike Pinecone (which is cloud-only and has restrictive free tier limits).

---

## 2. Embeddings: OpenAI CLIP (ViT-B/32)
**Chosen Model**: `openai/clip-vit-base-patch32` (via Hugging Face)
**Alternatives Considered**: `openai/clip-vit-large-patch14`, DinoV2, ResNet/EfficientNet features

### Why CLIP?
- **Joint Text-Image Space**: CLIP maps both images and text into a shared embedding space, allowing us to build a hybrid retrieval system (image-to-image or text-to-image) in the future.
- **Zero-Shot Transfer**: CLIP has strong zero-shot generalization capabilities across a wide range of image domains, including industrial defects, without explicit fine-tuning.
- **Compute Efficiency**: `clip-vit-base-patch32` produces 512-dimensional embeddings rapidly and runs extremely fast on standard CPUs or small GPUs, making it ideal for local testing and budget-friendly cloud instances.
- **Tradeoff vs DinoV2**: DinoV2 is superior for fine-grained patch-level visual similarity but does not have a native text encoder, making hybrid text-image querying harder to implement. CLIP is a more versatile baseline for multimodal RAG.

---

## 3. Vision-Language Model: Qwen2.5-VL
**Chosen Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
**Alternatives Considered**: LLaVA-1.5, Florence-2, Pixtral-12B

### Why Qwen2.5-VL?
- **State-of-the-Art Performance**: Qwen2.5-VL is currently one of the highest-performing open-source Vision-Language Models. It excels at visual reasoning, OCR, document understanding, and fine-grained visual comparison.
- **Dynamic Resolution support**: Qwen2.5-VL supports processing images at their native resolutions and aspect ratios, which is critical for identifying tiny defects in industrial parts (standard models downsample images to 224x224, destroying small defect features).
- **Scale options**: Available in 3B, 7B, and 72B parameter variants. The 3B model is extremely lightweight (running on ~6GB VRAM quantized), while the 7B version offers superior reasoning.
- **Tradeoff vs Florence-2**: Florence-2 is very fast and excellent at bounding box detection, but is less conversational and has weaker multi-image visual comparison capabilities compared to Qwen2.5-VL.

---

## 4. Backend API: FastAPI
**Chosen Tool**: FastAPI (Python)
**Alternatives Considered**: Flask, Django

### Why FastAPI?
- **Asynchronous Support**: Built on Starlette, FastAPI natively supports `async`/`await`, which is crucial for handling long-running AI inference requests without blocking the event loop.
- **Automatic Documentation**: Generates Swagger/OpenAPI docs automatically out of the box, making backend testing and API integration seamless.
- **Type Safety**: Uses Pydantic for request and response validation, reducing bugs and ensuring clean, self-documenting code.

---

## 5. Frontend UI: Gradio
**Chosen Tool**: Gradio
**Alternatives Considered**: Streamlit, React / Next.js

### Why Gradio?
- **Machine Learning Native**: Specifically built for ML prototyping. It supports multi-image inputs, side-by-side galleries, and chat interfaces with minimal boilerplate.
- **Integration**: Plugs directly into Python code, allowing swift integration with our FastAPI backend or embedding pipelines.
- **Interactivity**: Built-in support for displaying similarity scores alongside retrieved images, which is perfect for demonstrating RAG search quality.
- **Streamlit Comparison**: Streamlit is excellent for dashboarding, but Gradio is cleaner and more intuitive for direct model interaction and side-by-side multimodal inputs.
