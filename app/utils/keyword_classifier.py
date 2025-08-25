import re, json, numpy as np, pandas as pd
from app.config import DICT_TERMS

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer,util

# ------------------- Config -------------------
LABELS = ["Death", "Injury", "Device Malfunction"]

# thresholds (tune on your val set)
HI_THRESH = 0.9
MARGIN_THRESH = 0.10
ALPHA_DEFAULT = 0.60    # normal hybrid
ALPHA_LOW = 0.35        # when margin small / confidence low

EVIDENCE_TOPK = 12
EVIDENCE_MIN_COS = 0.30

# Death-specific safeguards
ENFORCE_DEATH_CONFIRMATION = True
DEATH_EVID_MIN = 0.8
DEATH_COS_MIN = 0.7
EVIDENCE_CAP_NO_STRICT_DEATH = 0.35


bert_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=bert_model)

# Strict death cues
DEATH_STRICT = [
    r"\bpatient (?:died|expired)\b", r"\bpronounced dead\b", r"\bdeclared dead\b",
    r"\bfound deceased\b", r"\bpassed away\b",
    r"\bfatal (?:event|outcome|complication)\b",
    r"\bdeath (?:occurred|reported|confirmed)\b",
    r"\bmortality\b", r"\bfatality\b"
]
NEGATION_PAT = r"(no|not|without|never|denies?|rule[sd]?\s*out)\s+(?:any\s+)?(death|died|deceased|fatal|expired)"

# ------------------- Helpers -------------------
def keybert_terms(text, top_k=15):
    kws = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1,3), stop_words='english', top_n=top_k
    )
    return [w for w,_ in kws]

def has_strict_death(text: str) -> bool:
    if re.search(NEGATION_PAT, text, flags=re.I):
        return False
    return any(re.search(pat, text, flags=re.I) for pat in DEATH_STRICT)

def parse_prob_dict(x):
    if isinstance(x, dict): 
        return x
    if isinstance(x, str):
        try:
            return json.loads(x.replace("'", '"'))
        except Exception:
            return {}
    return {}
def evidence_for_label(text, label):
    if label not in DICT_TERMS:
        print("here -----")
        return 0.0, []

    ext_terms = keybert_terms(text, top_k=EVIDENCE_TOPK)
    print("ext ---",ext_terms,label)
    if not ext_terms:
        print("here -----------")
        return 0.0, []

    dict_terms = DICT_TERMS[label]
    dict_emb = bert_model.encode(dict_terms, convert_to_tensor=True, normalize_embeddings=True)
    ext_emb  = bert_model.encode(ext_terms,  convert_to_tensor=True, normalize_embeddings=True)

    cos = util.cos_sim(ext_emb, dict_emb).cpu().numpy()

    matches = []
    min_cos = DEATH_COS_MIN if label == "Death" else EVIDENCE_MIN_COS

    for i, term in enumerate(ext_terms):
        j = int(cos[i].argmax())
        s = float(cos[i][j])
        print(s,min_cos,term,dict_terms[j])
        if s >= min_cos:
            matches.append({
                "extracted_term": term,
                "matched_dict_term": dict_terms[j],
                "score": round(s, 3)
            })

    # Always include fallback matches for Injury and Device Malfunction
    if label in ["Injury", "Device Malfunction"]:
        for i, term in enumerate(ext_terms):
            j = int(cos[i].argmax())
            s = float(cos[i][j])
            print("-----------")
            print(s,min_cos,term,dict_terms[j])
            if s >= 0.20 and not any(m["extracted_term"] == term for m in matches):
                matches.append({
                    "extracted_term": term,
                    "matched_dict_term": dict_terms[j],
                    "score": round(s, 3)
                })

    if not matches:
        if label == "Death" and has_strict_death(text):
            return 0.9, []
        return 0.0, []

    top_scores = sorted([m["score"] for m in matches], reverse=True)[:5]
    evid = float(np.mean(top_scores))

    if label == "Death":
        if has_strict_death(text):
            evid = max(evid, 0.9)
        else:
            evid = min(evid, EVIDENCE_CAP_NO_STRICT_DEATH)

    return evid, sorted(matches, key=lambda x: x["score"], reverse=True)



def clean_keyword_list(matches, topn=8, with_scores=False):
    seen, cleaned = set(), []
    for m in sorted(matches, key=lambda x: x["score"], reverse=True):
        term = m["matched_dict_term"]
        if term in seen:
            continue
        seen.add(term)
        if with_scores:
            cleaned.append(f"{term}:{m['score']}")
        else:
            cleaned.append(term)
        if len(cleaned) >= topn:
            break
    return cleaned

def pick_final_with_death_guard(text, combined, evidence):
    cand = max(combined.items(), key=lambda x: x[1])[0]
    if not ENFORCE_DEATH_CONFIRMATION or cand != "Death":
        return cand, "hybrid_fallback"
    strict = has_strict_death(text)
    evid_ok = evidence.get("Death", 0.0) >= DEATH_EVID_MIN
    if strict and evid_ok:
        return "Death", "hybrid_fallback_confirmed_death"
    non_death = {k: v for k, v in combined.items() if k != "Death"}
    alt = max(non_death.items(), key=lambda x: x[1])[0]
    return alt, "death_demoted_insufficient_evidence"



def classify_row(text, predicted_probability):
    prob = parse_prob_dict(predicted_probability)
    prob = {lab: float(prob.get(lab, 0.0)) for lab in LABELS}

    sorted_labs = sorted(LABELS, key=lambda l: prob[l], reverse=True)
    top1, top2 = sorted_labs[0], sorted_labs[1]
    p1, p2 = prob[top1], prob[top2]
    margin = p1 - p2

    high_conf_and_clear = (p1 >= HI_THRESH) and (margin > MARGIN_THRESH)
    labels_to_extract = [top1] if high_conf_and_clear else LABELS

    evidence = {lab: 0.0 for lab in LABELS}
    matches = {lab: [] for lab in LABELS}
    for lab in labels_to_extract:
        e, m = evidence_for_label(text, lab)
        evidence[lab], matches[lab] = e, m

    # ---- Decide final label ----
    if high_conf_and_clear:
        final_label = top1
        final_score = p1
        decision = "model_confident"
    else:
        alpha = ALPHA_LOW if margin <= MARGIN_THRESH else ALPHA_DEFAULT
        combined = {lab: alpha*prob[lab] + (1-alpha)*evidence[lab] for lab in LABELS}
        final_label, decision = pick_final_with_death_guard(text, combined, evidence)
        final_score = round(combined[final_label], 3)

    # ---- Build clean keyword columns ----
    flat_matches = {
        f"{lab.lower().replace(' ', '_')}_matches": json.dumps(matches[lab], ensure_ascii=False)
        for lab in LABELS
    }

    clean_cols = {}
    for lab in LABELS:
        terms_only = clean_keyword_list(matches.get(lab, []), topn=8, with_scores=False)
        terms_with_scores = clean_keyword_list(matches.get(lab, []), topn=8, with_scores=True)
        clean_cols[f"{lab.lower().replace(' ', '_')}_keywords"] = "; ".join(terms_only)
        clean_cols[f"{lab.lower().replace(' ', '_')}_keywords_scored"] = "; ".join(terms_with_scores)

    # ---- Extracted keywords directly from KeyBERT ----
    extracted_keywords = keybert_terms(text, top_k=10)

    # ---- Build explanation ----
    if decision == "model_confident":
        explanation = f"Predicted '{final_label}' with high confidence ({round(final_score,3)})."
    elif decision == "hybrid_fallback":
        explanation = (
            f"Label '{final_label}' chosen by combining model prediction "
            f"and evidence from matched keywords."
        )
    elif decision == "hybrid_fallback_confirmed_death":
        explanation = (
            "Death label confirmed due to strong textual cues and matching keywords."
        )
    elif decision == "death_demoted_insufficient_evidence":
        explanation = (
            "Death label downgraded due to lack of strong evidence, assigned alternate label."
        )
    else:
        explanation = f"Final label '{final_label}' based on combined model + evidence."

    return {
        "final_label": final_label,
        "final_score": final_score,
        "decision": decision,
        "explanation": explanation,
        "extracted_keywords": extracted_keywords,
        "p_model": json.dumps({lab: round(prob[lab], 3) for lab in LABELS}),
        "evidence": json.dumps({lab: round(evidence[lab], 3) for lab in LABELS}),
        **flat_matches,
        **clean_cols
    }

def classify_with_keywords(report: str, model_probs: dict):
    """Combines model probabilities + evidence + extracted keywords + explanation."""
    return classify_row(report, model_probs)
