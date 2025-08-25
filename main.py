

import gradio as gr
from app.agents.crew_pipeline import run_pipeline

def process_report(report: str):
    try:
        result = run_pipeline(report)
        output_text = f"""
### ** Model Prediction**
**Label:** {result['model_prediction']}
**Probabilities:** {result['model_probabilities']}
---
### ** Final Classification**
**Label:** {result['final_classification']}
**Decision Rule:** {result['decision']}
---
###  **Extracted Keywords**
- **Death:** {result['keywords']['death'] or "None"}
- **Injury:** {result['keywords']['injury'] or "None"}
- **Device Malfunction:** {result['keywords']['device_malfunction'] or "None"}

### 📖 **Explanation**
{result['summary']}
"""
        return output_text
    except Exception as e:
        return f" Error: {e}"

with gr.Blocks() as demo:
    gr.Markdown("## **Adverse Event Detection** 🩺")
    gr.Markdown("Paste an adverse event report below and get classification, extracted keywords, similarity scores, and an explanation.")

    with gr.Row():
        with gr.Column():
            report_input = gr.Textbox(
                label="Adverse Event Report",
                placeholder="Paste report text here...",
                lines=8
            )
            submit_btn = gr.Button("Analyze Report")
        with gr.Column():
            output_box = gr.Markdown()

    submit_btn.click(fn=process_report, inputs=report_input, outputs=output_box)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


# ---

# ###  **Similarity Scores**
# - **Death Scores:** {result['keywords_scored']['death'] or "None"}
# - **Injury Scores:** {result['keywords_scored']['injury'] or "None"}
# - **Device Malfunction Scores:** {result['keywords_scored']['device_malfunction'] or "None"}

# ---