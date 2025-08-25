

from crewai import Agent, Task, Crew
from app.utils.model_file import predict_ae_type
from app.utils.keyword_classifier import classify_with_keywords
from langchain_community.llms import Ollama  
from langchain.prompts import PromptTemplate

# Configure Ollama (Mistral)
ollama_llm = Ollama(model="mistral:latest", temperature=0.3)

def generate_user_response(report, classification_result):
    """
    Generates a user-friendly explanation of why the classification was chosen.
    Includes extracted report terms, matched dictionary terms, and evidence scores.
    """

    #Dynamically pick keywords based on the final predicted label
    final_label = classification_result["final_label"].lower().replace(" ", "_")
    keywords = classification_result.get(f"{final_label}_keywords") or "No significant keywords found"
    keywords_scored = classification_result.get(f"{final_label}_keywords_scored") or "No significant scores available"

    # Build response prompt including extracted + matched keywords + scores
    prompt = PromptTemplate(
        input_variables=["report", "label", "keywords", "scores"],
        template="""
        Based on the following adverse event report:

        Report: {report}

        Final Classification: {label}

        Extracted Keywords & Matched Dictionary Terms:
        {keywords}

        Similarity Scores:
        {scores}

        Please explain concisely why this classification was chosen,
        highlighting the specific report terms that contributed to the decision.
        """
    )

    chain = prompt | ollama_llm
    return chain.invoke({
        "report": report,
        "label": classification_result["final_label"],
        "keywords": keywords,
        "scores": keywords_scored,
    })

def run_pipeline(report: str):
    #  Model prediction + probabilities
    predicted_label, probs = predict_ae_type(report)

    # Keyword-based classification
    classification_result = classify_with_keywords(report, probs)

    #  Guarantee final_label is always available
    if not classification_result.get("final_label"):
        classification_result["final_label"] = predicted_label or "Unclassified"

    # Generate explanation
    summary = generate_user_response(report, classification_result)

    #  Return structured output
    return {
        "model_prediction": predicted_label,
        "model_probabilities": probs,
        "final_classification": classification_result["final_label"],
        "decision": classification_result.get("decision", "Model prediction used"),
        "keywords": {
            "death": classification_result.get("death_keywords", ""),
            "injury": classification_result.get("injury_keywords", ""),
            "device_malfunction": classification_result.get("device_malfunction_keywords", "")
        },
        "keywords_scored": {
            "death": classification_result.get("death_keywords_scored", ""),
            "injury": classification_result.get("injury_keywords_scored", ""),
            "device_malfunction": classification_result.get("device_malfunction_keywords_scored", "")
        },
        "summary": summary
    }
