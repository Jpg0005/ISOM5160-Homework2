"""
ISOM5160 Homework 2 — SMS Spam / Smishing Alert App
Course: ISOM5160-L3 Python Programming for Business Applications, Fall 2026 (PG ICSM)

INSTRUCTIONS (read carefully)
- Complete the marked STUDENT sections and the function bodies below.
- Keep these function names: load_model, load_test_samples, classify_text,
  compute_accuracy, main.
- Call main() at the bottom of this file (the starter already does this).
- In GitHub and on Streamlit Cloud, this file must be named app.py.
- Submit to Canvas: a zip file containing app.py, requirements.txt, and
  Homework2_<StudentID>_link.txt (Streamlit Cloud URL and GitHub URL).
  Put those three files at the root of the zip, not inside a subfolder.
- Do NOT fine-tune a model. Do NOT scrape the web.
- Do NOT change DATASET_ID, MODEL_ID, N_TEST, or SPLIT_SEED.
- When you deploy, open Advanced settings and set Python to 3.12.
  If the app is already deployed, delete it and redeploy — the Python
  version cannot be changed in place.
- If the app calls a Hugging Face model, store a fine-grained Hugging Face
  API token in Streamlit Cloud secrets as HF_TOKEN. Do not put the token
  in app.py, requirements.txt, or any public file.
- First run may download the model and dataset (needs network). CPU is OK.

SECRET PHRASE (must appear on the app page): ISOM5160-HW2-SMS
"""

# =============================================================================
# STUDENT ANSWERS BLOCK (fill in; keep this in the SAME .py file)
# =============================================================================
# Student ID: 21318735
#
# GenAI acknowledgment (required if you used GenAI; otherwise write "None"):
#   - Tools used: OpenAI Codex
#   - How you used them: Reviewed the requirements, implemented and tested the
#     starter code, and helped draft the business answers.
#
# Q1. Who is the primary business user of this app?
#     (one short sentence; e.g., an employee or a security analyst reviewing
#      inbound SMS for spam or scam-like messages)
#   Answer: The primary user is a security analyst reviewing suspicious SMS
#   messages reported by employees.
#
# Q2. Look at the accuracy from your 100-sample test. Is accuracy enough
#     for a business to rely on this app, and why or why not?
#     2–4 short sentences max. You MUST cite the accuracy figure from
#     YOUR app run. Comment on the cost of a false positive (blocking a
#     legitimate SMS, such as a bank or one-time code message) versus a
#     false negative (missing a spam or scam-like message).
#   My 100-message test achieved 99.00% accuracy and missed one spam message.
#   This accuracy alone is insufficient for a business to rely on the app.
#   A false positive can block a bank alert or one-time code, while a false
#   negative can expose an employee to a scam.
#
# =============================================================================

import streamlit as st
from datasets import load_dataset
from transformers.pipelines import pipeline

# Locked by the instructor — do not change these four values.
DATASET_ID = "ucirvine/sms_spam"
MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
N_TEST = 100
SPLIT_SEED = 42

# Course identity (required)
STUDENT_ID = "21318735"
COURSE_SECRET = "ISOM5160-HW2-SMS"  # do not change


def map_label(raw_label):
    """Map model or dataset labels to the two display names used in this app."""
    text = str(raw_label).strip().lower()
    if text in {"ham", "legitimate", "label_0", "0"}:
        return "legitimate"
    if text in {"spam", "smishing", "label_1", "1"}:
        return "spam"
    return str(raw_label)


@st.cache_resource
def load_model(token):
    """Load the SMS text-classification model once and reuse it."""
    # This function is provided. Keep device="cpu" and
    # model_kwargs={"low_cpu_mem_usage": False} so Streamlit Cloud can load
    # the model on CPU.
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        token=token,
        device="cpu",
        truncation=True,
        max_length=512,
        model_kwargs={"low_cpu_mem_usage": False},
    )
    return classifier


@st.cache_data
def load_test_samples():
    """
    Load the Hugging Face dataset and take a fixed set of 100 samples.

    The Hub copy has only a train split, so this homework holds out 100
    rows after a shuffle with SPLIT_SEED. Do not change the seed or N_TEST,
    so every student is tested on the same 100 messages.
    """
    raw = load_dataset(DATASET_ID, split="train")
    picked = raw.shuffle(seed=SPLIT_SEED).select(range(N_TEST))
    texts = list(picked["sms"])
    true_labels = [map_label(label) for label in picked["label"]]
    return texts, true_labels


def classify_text(classifier, text):
    """Return (display_label, confidence) for one SMS string."""
    result = classifier(text)[0]
    return map_label(result["label"]), float(result["score"])


def compute_accuracy(pred_labels, true_labels):
    """Return accuracy as a float between 0 and 1."""
    matches = sum(
        prediction == truth
        for prediction, truth in zip(pred_labels, true_labels, strict=True)
    )
    return matches / len(true_labels) if len(true_labels) else 0.0


def main():
    st.title("ISOM5160")
    st.write("Homework 2 — SMS Spam / Smishing Alert")
    st.write(f"**Student ID** {STUDENT_ID} · **Course secret** `{COURSE_SECRET}`")

    # Use a Hugging Face fine-grained token from Streamlit Cloud secrets
    # if one is set. Do not put the token in this file.
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except FileNotFoundError:
        hf_token = None
    classifier = load_model(hf_token)

    alert_text = st.text_area(
        "SMS message",
        value=(
            "WINNER!! You have been selected to receive a $900 prize. "
            "Claim now at http://prize-alert.example or the offer expires."
        ),
        height=140,
    )

    if st.button("Classify SMS"):
        message = alert_text.strip()
        if message:
            label, confidence = classify_text(classifier, message)
            st.write(f"**Prediction** {label} · **Confidence** {confidence:.2%}")
        else:
            st.warning("Enter an SMS message to classify.")

    st.subheader("Test on 100 Hugging Face samples")
    st.write(
        "Click the button to load 100 messages from the Hugging Face "
        "dataset and measure accuracy."
    )
    if st.button("Run 100-sample test"):
        with st.spinner("Classifying 100 SMS messages…"):
            texts, true_labels = load_test_samples()
            results = []
            for index, (text, truth) in enumerate(zip(texts, true_labels, strict=True), 1):
                prediction, confidence = classify_text(classifier, text)
                preview = " ".join(text.split())
                if len(preview) > 100:
                    preview = preview[:97] + "..."
                results.append({
                    "#": index,
                    "Text (short)": preview,
                    "Ground truth": truth,
                    "Prediction": prediction,
                    "Confidence": f"{confidence:.2%}",
                    "Match": prediction == truth,
                })

            pred_labels = [row["Prediction"] for row in results]
            accuracy = compute_accuracy(pred_labels, true_labels)

        st.metric("Accuracy", f"{accuracy:.2%}")
        st.dataframe(results, hide_index=True)


if __name__ == "__main__":
    main()
