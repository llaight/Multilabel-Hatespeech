#SVM
import streamlit as st
import pickle
import re
import numpy as np
from typing import List
from v2_multilabel_rulebased_hate_svm_sgd_mbert import (
    hybrid_predict,
    load_stopwords,
    setup_nltk,
    load_lemma_dict,
    clean_text,
    RULE_LEXICONS,
    LABEL_COLUMNS,
    OneVsRestSVM,
    LinearSVM,

)

st.set_page_config(
    page_title="Multilabel Hate Speech Detection",
    page_icon="📊",
    layout="wide"
)

st.title('SVM-Hybrid Model Multilabel Hate Speech Detection')
st.write('Enter a text to detect hate speech:')
user_input = st.text_area(
    label="Hate Text",
    help="Enter a text to detect hate speech"
)

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

def boosted_probabilities(
    norm_probs: np.ndarray, 
    min_share: float = 0.51, 
    max_share: float = 0.85,
    validate_input: bool = True
) -> np.ndarray:

    if validate_input:
        if not isinstance(norm_probs, np.ndarray):
            norm_probs = np.array(norm_probs)
        
        if len(norm_probs) == 0:
            raise ValueError("Input probability array cannot be empty")
        
        if np.any(norm_probs < 0):
            raise ValueError("Probabilities cannot be negative")
        
        prob_sum = np.sum(norm_probs)
        if not np.isclose(prob_sum, 1.0, rtol=1e-5):
            print(f"Warning: Probabilities sum to {prob_sum:.6f}, normalizing...")
            norm_probs = norm_probs / prob_sum
        
        if not (0 < min_share <= max_share <= 1.0):
            raise ValueError("Must have 0 < min_share <= max_share <= 1.0")
    
    # Handle edge cases
    if len(norm_probs) == 1:
        return np.array([1.0])
    
    # Find the class with maximum probability
    max_idx = np.argmax(norm_probs)

    sorted_probs = np.sort(norm_probs)
    gap = norm_probs[max_idx] - sorted_probs[-2]
    
    max_possible_gap = norm_probs[max_idx]
    if max_possible_gap > 0:
        gap_ratio = gap / max_possible_gap
    else:
        gap_ratio = 0
    
    boost = min_share + (max_share - min_share) * gap_ratio
    boost = np.clip(boost, min_share, max_share)
    
    # Calculate sum of other probabilities
    other_sum = np.sum(np.delete(norm_probs, max_idx))
    
    # Create boosted probability array
    boosted = np.zeros_like(norm_probs, dtype=np.float64)
    
    if other_sum == 0:
        # Edge case: only one non-zero probability
        boosted[max_idx] = 1.0
    else:
        # Assign boost to max class
        boosted[max_idx] = boost
        
        # Redistribute remaining probability proportionally among other classes
        remaining_prob = 1.0 - boost
        for i in range(len(norm_probs)):
            if i != max_idx:
                boosted[i] = remaining_prob * (norm_probs[i] / other_sum)
    
    return boosted

# Always show the progress bars
st.write("Prediction (per label):")

color_map = {
    "Age": "#3498db",
    "Gender": "#9b59b6",
    "Physical": "#e67e22",
    "Race": "#e74c3c",
    "Religion": "#16a085",
    "Politics": "#f1c40f",
    "Others": "#34495e"
}

if st.button('Predict'):
    if user_input == "":
        st.warning("Please enter a text to classify.")
        # Show zeroed bars if no input
        for label in LABEL_COLUMNS:
            st.write(f"{label}: (probability: 0.00)")
            color= color_map.get(label, '#1abc9c')
            st.markdown(
                f"""
                <div style="background: #eee; border-radius: 5px; height: 20px; width: 100%; margin-bottom:8px;">
                    <div style="background: {color}; width: 0%; height: 100%; border-radius: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
        )
    elif len(user_input.split()) < 3:
        st.warning("Please enter at least 3 words.")
        for label in LABEL_COLUMNS:
            st.write(f"{label}: (probability: 0.00)")
            color= color_map.get(label, '#1abc9c')
            st.markdown(
                f"""
                <div style="background: #eee; border-radius: 5px; height: 20px; width: 100%; margin-bottom:8px;">
                    <div style="background: {color}; width: 0%; height: 100%; border-radius: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
        )

    else:
        # Load model and vectorizer
        with open('pages/svm_model.pkl', 'rb') as model_file:
            model_bundle = pickle.load(model_file)

        model = model_bundle['svm_model']
        vectorizer = model_bundle['vectorizer']

        # Setup preprocessing
        setup_nltk()
        all_stopwords = load_stopwords()
        tagalog_lemma = load_lemma_dict('filipino_lemma.txt')
        from nltk.stem import WordNetLemmatizer
        eng_lemma = WordNetLemmatizer()

        # Clean and prepare input
        cleaned = clean_text(user_input, all_stopwords, tagalog_lemma, eng_lemma)
        feats = vectorizer.transform([user_input]).astype(np.float32).toarray()
        decision_values = model.decision_function(feats)[0]
        
        probas = 1 / (1 + np.exp(-decision_values))

        total = np.sum(probas)
        if total == 0:
            norm_probs = np.zeros_like(probas)
        else:
            norm_probs = probas / total


        boosted_probs = boosted_probabilities(norm_probs, min_share=0.51, max_share=0.85)

        for label, value, prob in zip(LABEL_COLUMNS, (boosted_probs >= 0.5).astype(int), boosted_probs):
            color = color_map.get(label, '#1abc9c')
            st.write(f"{label}: (probability: {prob*100:.2f})%")
            st.markdown(
                f"""
                <div style="background: #eee; border-radius: 5px; height: 20px; width: 100%; margin-bottom:8px;">
                    <div style="background: {color}; width: {prob*100:.2f}%; height: 100%; border-radius: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        
else:
    # Show zeroed bars before prediction
    for label in LABEL_COLUMNS:
        st.write(f"{label}: (probability: 0.00)")
        color= color_map.get(label, '#1abc9c')
        st.markdown(
            f"""
            <div style="background: #eee; border-radius: 5px; height: 20px; width: 100%; margin-bottom:8px;">
                <div style="background: {color}; width: 0%; height: 100%; border-radius: 5px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
       )
