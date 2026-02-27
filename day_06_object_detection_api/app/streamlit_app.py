"""
Day 06 — Object Detection API: Streamlit Demo UI
Interactive object detection interface with image upload and visualization.
"""

import io
import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="🔍 Object Detection — YOLOv8",
    page_icon="🔍",
    layout="wide",
)


def check_health() -> dict | None:
    """Check if the API is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def main():
    st.title("🔍 Object Detection with YOLOv8")
    st.markdown(
        "Upload an image or provide a URL to detect objects in real-time "
        "using **YOLOv8** — the state-of-the-art object detection model."
    )
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Health check
        health = check_health()
        if health:
            status_color = "🟢" if health.get("status") == "healthy" else "🟡"
            st.success(f"{status_color} API Connected")
            st.metric("Model", health.get("model_name", "N/A"))
            st.metric("Classes", health.get("num_classes", 0))
        else:
            st.error("🔴 API not available. Start with: `uvicorn api.main:app`")
            st.stop()

        st.markdown("---")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score to display a detection.",
        )

        iou_threshold = st.slider(
            "IoU Threshold (NMS)",
            min_value=0.05,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for Non-Maximum Suppression.",
        )

        max_detections = st.slider(
            "Max Detections",
            min_value=1,
            max_value=300,
            value=100,
            step=10,
        )

        st.markdown("---")
        show_table = st.checkbox("Show detections table", value=True)
        show_annotated = st.checkbox("Show annotated image", value=True)

        st.markdown("---")
        st.header("📋 Supported Classes")
        try:
            resp = requests.get(f"{API_URL}/classes", timeout=5)
            if resp.status_code == 200:
                classes = resp.json().get("classes", [])
                with st.expander(f"View all {len(classes)} classes"):
                    for i, cls in enumerate(classes):
                        st.text(f"{i:3d}: {cls}")
        except Exception:
            st.caption("Could not load class list.")

    # ─── Main Content ────────────────────────────────────────────────
    tab_upload, tab_url = st.tabs(["📤 Upload Image", "🌐 Image URL"])

    # ── Tab 1: Upload ────────────────────────────────────────────────
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported: JPG, PNG, BMP, WebP",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            col_orig, col_result = st.columns(2)

            with col_orig:
                st.subheader("📷 Original Image")
                st.image(image, use_container_width=True)

            # Run detection
            with st.spinner("🔍 Detecting objects..."):
                # Get JSON results
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {
                    "confidence": confidence,
                    "iou_threshold": iou_threshold,
                    "max_detections": max_detections,
                }

                resp = requests.post(
                    f"{API_URL}/detect",
                    files=files,
                    params=params,
                    timeout=60,
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Get annotated image
                    if show_annotated:
                        uploaded_file.seek(0)
                        files_ann = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        resp_ann = requests.post(
                            f"{API_URL}/detect/annotate",
                            files=files_ann,
                            params=params,
                            timeout=60,
                        )

                        with col_result:
                            st.subheader("🎯 Detections")
                            if resp_ann.status_code == 200:
                                annotated_img = Image.open(io.BytesIO(resp_ann.content))
                                st.image(annotated_img, use_container_width=True)
                            else:
                                st.warning("Could not generate annotated image.")

                    # Metrics
                    st.markdown("---")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Objects Detected", result["num_detections"])
                    with col_b:
                        st.metric("Unique Classes", len(result.get("class_counts", {})))
                    with col_c:
                        st.metric("Inference Time", f"{result['elapsed_ms']:.0f}ms")
                    with col_d:
                        size = result.get("image_size", {})
                        st.metric("Image Size", f"{size.get('width', 0)}x{size.get('height', 0)}")

                    # Class breakdown
                    if result.get("class_counts"):
                        st.markdown("### 📊 Class Distribution")
                        import pandas as pd
                        df_classes = pd.DataFrame(
                            list(result["class_counts"].items()),
                            columns=["Class", "Count"],
                        ).sort_values("Count", ascending=False)
                        st.bar_chart(df_classes.set_index("Class"))

                    # Detections table
                    if show_table and result["detections"]:
                        st.markdown("### 📋 Detection Details")
                        import pandas as pd
                        rows = []
                        for det in result["detections"]:
                            rows.append({
                                "Class": det["class_name"],
                                "Confidence": f"{det['confidence']:.1%}",
                                "X1": det["bbox"]["x1"],
                                "Y1": det["bbox"]["y1"],
                                "X2": det["bbox"]["x2"],
                                "Y2": det["bbox"]["y2"],
                                "Area": det["area"],
                            })
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.error(f"Detection failed: {resp.text}")

    # ── Tab 2: URL ───────────────────────────────────────────────────
    with tab_url:
        url = st.text_input(
            "Image URL",
            placeholder="https://example.com/photo.jpg",
        )

        if url and st.button("🔍 Detect from URL", type="primary"):
            with st.spinner("Downloading and detecting..."):
                try:
                    payload = {
                        "url": url,
                        "confidence": confidence,
                        "iou_threshold": iou_threshold,
                        "max_detections": max_detections,
                    }
                    resp = requests.post(
                        f"{API_URL}/detect/url",
                        json=payload,
                        timeout=60,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"✅ Detected {result['num_detections']} objects")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Objects", result["num_detections"])
                        with col_b:
                            st.metric("Classes", len(result.get("class_counts", {})))
                        with col_c:
                            st.metric("Time", f"{result['elapsed_ms']:.0f}ms")

                        if result["detections"]:
                            st.markdown("### Detections")
                            for det in result["detections"]:
                                st.markdown(
                                    f"- **{det['class_name']}** — "
                                    f"{det['confidence']:.1%} confidence"
                                )
                    else:
                        st.error(f"Error: {resp.text}")

                except Exception as e:
                    st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.caption(
        "Day 06 — Object Detection API | "
        "YOLOv8 + FastAPI + Streamlit"
    )


if __name__ == "__main__":
    main()
