
# **Adverse Event Classification & Explainable AI Pipeline**
> **Powered by CrewAI · Fine-Tuned Phi-3 Mini Instruct · Hybrid Evidence + Keyword Extraction**

---

## ** Overview**

This project is an **end-to-end explainable AI pipeline** for **classifying adverse events** reported in medical device FOI texts. It integrates a **fine-tuned Phi-3 Mini Instruct model** with a **keyword evidence layer** and **CrewAI-based orchestration** to deliver **accurate predictions** with **transparent explanations**.The fined tuned model is focused on the reports related to vascular closure devices.

Unlike traditional APIs where you hit a single endpoint, **CrewAI** organizes the process into **specialized AI agents** that **collaborate autonomously** to produce structured, human-friendly outputs.

---

## ** Features**

1. **Fine-Tuned Model** — Uses **Phi-3 Mini Instruct**, fine-tuned via **LoRA** for FDA-style adverse event classification.  
2. **Hybrid Classification** — Combines **model probabilities** + **keyword dictionary evidence** for robust results.  
3. **Explainable Result ** — Generates a **clear, user-friendly explanation** of *why* the classification was chosen.  
4. **Keyword Extraction Layer** — Uses **KeyBERT** to extract **high-similarity terms** and **matched evidence**.  
5. **CrewAI Orchestration** — Coordinates model prediction, keyword extraction, and LLM-based summarization seamlessly.  
6. **Structured Outputs** — Returns JSON with classification, extracted keywords, similarity scores, and explanations.

---

## ** Why CrewAI Instead of FastAPI + Sequential API Calls?**

Traditionally, you'd build a **FastAPI service** where:
1. **Call 1:** Predict using the fine-tuned Phi-3 model.
2. **Call 2:** Extract keywords.
3. **Call 3:** Generate explanations.

However, this leads to **tight coupling** and **hard-to-scale logic**.

With **CrewAI**, we define **specialized agents** and their roles:  

- **Classifier Agent** → Runs the fine-tuned Phi-3 Mini Instruct model to predict adverse event types.  
- **Keyword Agent** → Uses KeyBERT + custom medical dictionaries to extract strong evidence.  
- **Explanation Agent** → Uses Mistral/Ollama to generate user-friendly reasoning based on prediction + evidence.  

These agents **communicate internally** within CrewAI, reducing API round trips and ensuring **better context sharing** → **faster** + **smarter** responses.

---

## ** System Architecture**

### **Workflow Diagram**

```mermaid
flowchart TD
    A[User Inputs FOI Report] --> B[Classifier Agent (Phi-3 Mini + LoRA)]
    B --> C[Keyword Agent (KeyBERT + Custom AE Dictionary)]
    C --> D[Explanation Agent (Mistral via Ollama)]
    D --> E[Structured JSON Output]

```
---

## ** Tech Stack**

| Component            | Technology Used             | Purpose                                    |
|----------------------|-----------------------------|------------------------------------------|
| **Model**           | Phi-3 Mini Instruct (LoRA)   | Fine-tuned for adverse event detection  |
| **Keyword Layer**   | KeyBERT + Custom Dictionary  | Extract keywords & match known terms   |
| **Orchestration**   | CrewAI                       | Multi-agent task coordination           |
| **LLM Reasoning**   | Mistral / Ollama             | Generate user-friendly explanations    |
| **Vector Search**   | Optional Qdrant/Chroma       | Future-proof semantic evidence retrieval |
| **Frontend**        | Gradio / Streamlit (optional)| Simple user interface                  |

---

## ** Model Fine-Tuning (Phi-3 + LoRA)**

We fine-tuned **Phi-3 Mini Instruct** (~3.8B params) using **LoRA adapters** for **low-resource, domain-specific training**.  

- **Dataset** → FDA MAUDE adverse event reports (2021–2024)  
- **Labels** → `Device Malfunction`, `Injury`, `Death`  
- **Technique** → LoRA fine-tuning to minimize compute cost  
- **Output** → A lightweight model optimized for **adverse event type detection**.

This model achieves **higher recall on rare events** (like `Death`) compared to zero-shot models.

---

## ** Keyword Extraction Layer**

To make results **auditable** and **transparent**, we integrate a **dictionary + embedding-based** evidence layer:

1. **Extract top-N keywords** from FOI_TEXT using **KeyBERT**.
2. **Match extracted terms** against a curated **medical dictionary** (occlusion alarms, flow blocked, device failure, etc.).
3. **Compute cosine similarity scores** between report embeddings and known adverse event keywords.
4. **Feed matched terms + scores** into the explanation agent.

This ensures the **final classification is interpretable** and **medically meaningful**.

---

## ** CrewAI Pipeline**

### **Step 1: Model Prediction**
- The **Classifier Agent** invokes the fine-tuned Phi-3 model.
- Outputs label + probabilities.

### **Step 2: Keyword Evidence**
- The **Keyword Agent** extracts **strong evidence terms** from the report.
- Matches against a curated adverse event dictionary.

### **Step 3: Generate Explanation**
- The **Explanation Agent** uses **Mistral via Ollama**.
- Produces a **concise, user-friendly summary**.

---

## ** Installation**

---

## **▶ Usage**

### **Run the CrewAI Pipeline**

```python
from app.agents.crew_pipeline import run_pipeline

report = """
The pump stopped suddenly during infusion, triggering occlusion alarms.
Patient experienced temporary discomfort but recovered after device reset.
"""

result = run_pipeline(report)
print(result)
```

**Sample Output**:
```json
{
  "model_prediction": "Device Malfunction",
  "model_probabilities": {
    "Device Malfunction": 0.87,
    "Injury": 0.10,
    "Death": 0.03
  },
  "final_classification": "Device Malfunction",
  "keywords": ["pump stopped", "occlusion alarm", "flow blocked"],
  "keywords_scored": ["pump stopped (0.94)", "occlusion alarm (0.91)"],
  "summary": "The device likely malfunctioned due to pump stoppage and occlusion alarms, but the patient recovered after device reset."
}
```

---

## ** Advantages**

- **Explainable Results** → Combines LLM predictions with **visible evidence**.
- **Scalable** → Agents can be replaced or extended independently.
- **Cost-Efficient** → LoRA fine-tuning on Phi-3 reduces GPU memory needs.
- **Transparent** → Matched keywords + similarity scores enhance auditability.

---

## ** Future Enhancements**
- Instead of classification bring in the explanation/justification as well from fine tuned model.
- Integrate **vector database (Qdrant)** for **semantic context retrieval**.
- Add **multilingual support** for FOI_TEXT in other regions.
- Deploy **streaming Gradio UI** for real-time feedback.


---
## 📚 References

1. **FDA MAUDE Database** – Adverse Event Report Data  
   [https://www.fda.gov/medical-devices/medical-device-reporting-mdr-how-report-medical-device-problems/adverse-event-reporting-data-files]

2. **Phi-3 Mini (Microsoft)** – Lightweight fine-tuned LLM for classification  
   [https://huggingface.co/microsoft/phi-3-mini-4k-instruct]

---
##  License

MIT License © 2025 [Shivangi Bhave](https://github.com/Sbhave01)

