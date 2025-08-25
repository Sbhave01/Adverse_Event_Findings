# app/utils/model_file.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from torch.nn.functional import softmax
import torch

label_list = ["Death", "Injury", "Device Malfunction"]
model_name = "microsoft/Phi-3-mini-4k-instruct"

compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Load model once
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, "app/utils/trained-model")
model.eval()
model.config.use_cache = False

def predict_ae_type(report: str):
    prompt = f"Classify the type of adverse event as Death, Injury, or Device Malfunction.\n\nEvent: {report}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    prediction = decoded.split("Answer:")[-1].strip().split("\n")[0]

    # Probabilities
    transition_scores = outputs.scores
    probs = {}
    if transition_scores:
        step_logits = transition_scores[0]
        step_probs = softmax(step_logits, dim=-1)
        for label in label_list:
            label_token_id = tokenizer(label, add_special_tokens=False).input_ids[0]
            probs[label] = step_probs[0, label_token_id].item()
    else:
        probs = {label: None for label in label_list}

    return prediction, probs
