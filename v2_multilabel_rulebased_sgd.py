import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle


# Configuration
CSV_FILE = Path("Multilabel-Hatespeech/Annotate _ Datasets - Annotation v3 FINAL NA BEH.csv")
TEXT_COLUMN = "Text"
LABEL_COLUMNS = [
    "Age",
    "Gender",
    "Physical",
    "Race",
    "Religion",
    "Politics",
    "Others",
]
MAX_FEATS = 5_000
NGRAM_RANGE = (1, 2)
UNCERTAINTY_THRESHOLD = 0.05

# Rule-based lexicons
RULE_LEXICONS = {
    "Age": ["tanda", "boomer", "bata", "matanda", "gurang", "genz", "millenial"],
    "Gender": ["bakla", "tomboy", "babae kasi", "bading", "gay", "lesbian"],
    "Physical": ["pangit", "pandak", "tabatsoy", "baldado"],
    "Race": ["chinese", "unggoy", "nigger", "chingchong"],
    "Religion": ["iglesia", "muslim", "terrorist", "jihad", "infidel", "catholic", "pari"],
    "Politics": ["bbm", "dilawan", "kakampink", "komunista", "presidente", "lugaw"],
    "Others": ["tanga", "ulol", "gago", "bobo", "stupido", "putangina mo", "putangina", "tangina"],
}


class SGD:
    def __init__(self, n_features: int, C: float = 10.0, lr: float = 0.1, epochs: int = 150, seed: int = 42):
        self.C = C
        self.lr0 = lr
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_signed = np.where(y > 0, 1.0, -1.0).astype(np.float32)
        n_samples = X.shape[0]
        for epoch in range(1, self.epochs + 1):
            indices = self.rng.permutation(n_samples)
            eta = self.lr0 / (1 + epoch * 0.01)
            for i in indices:
                xi = X[i]
                yi = y_signed[i]
                margin = yi * (np.dot(self.w, xi) + self.b)

                # L2 regularization term
                reg_term = eta / (self.C * n_samples)

                if margin < 1:
                    # Apply regularization and hinge loss gradient
                    self.w = (1 - reg_term) * self.w + eta * self.C * yi * xi
                    self.b += eta * self.C * yi
                else:
                    # Only apply regularization
                    self.w = (1 - reg_term) * self.w

        # bias centering fix
        self.b = -np.mean(X @ self.w)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(np.int8)

class OneVsRestSGD:
    def __init__(self, n_labels: int, n_features: int, **sgd_kwargs):
        self.n_labels = n_labels
        self.models = [SGD(n_features, **sgd_kwargs) for _ in range(n_labels)]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        for idx in range(self.n_labels):
            self.models[idx].fit(X, Y[:, idx])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([m.decision_function(X) for m in self.models])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(np.int8)

def setup_nltk():
    """Download required NLTK data."""
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def load_stopwords():
    """Load English and Tagalog stopwords."""
    english_stopwords = set(stopwords.words('english'))
    
    tagalog_stopwords = set([
        "akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay","bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin","gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag","iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas","ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino","kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad","kaya","kaysa","ko","kong","kulang","kumuha","kung","laban","lahat","lamang","likod","lima","maaari","maaaring","maging","mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na","nabanggit","naging","nagkaroon","nais","nakita","namin","napaka","narito","nasaan","ng","ngayon","ni","nila","nilang","nito","niya","niyang","noon","o","pa","paano","pababa","paggawa","pagitan","pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas","pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino","siya","tatlo","tayo","tulad","tungkol","una","walang"
    ])
    
    return english_stopwords.union(tagalog_stopwords)


def load_lemma_dict(file_path):
    """Load Tagalog lemmatization dictionary."""
    lemma_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                word, lemma = line.strip().split(':')
                lemma_dict[word.strip()] = lemma.strip()
    return lemma_dict


def clean_text(text: str, all_stopwords: set, tagalog_lemma: dict, eng_lemma: WordNetLemmatizer) -> str:
    """Clean and preprocess text."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in all_stopwords]
    # Lemmatization
    lemmatized_tokens = []
    for t in tokens:
        lemma = tagalog_lemma.get(t)
        if lemma is not None:
            lemmatized_tokens.append(lemma)
        else:
            lemmatized_tokens.append(eng_lemma.lemmatize(t))
    return " ".join(lemmatized_tokens)


def tokenize(text: str) -> List[str]:
    """Tokenize text."""
    return re.findall(r"\b\w+\b", text.lower())


def rule_based_labels(cleaned_text: str, lexicons: dict, label_order: List[str]) -> List[int]:
    """Apply rule-based labeling using lexicons."""
    labels = [0] * len(label_order)
    for i, label in enumerate(label_order):
        for kw in lexicons[label]:
            if kw in cleaned_text:
                labels[i] = 1
                break
    return labels


def hybrid_predict(
    raw_texts,
    vec,
    model,
    lexicons,
    label_order,
    all_stopwords,
    tagalog_lemma,
    eng_lemma,
    thr: float = UNCERTAINTY_THRESHOLD,
):
    """Hybrid prediction using ML model and rule-based approach."""
    feats = vec.transform(raw_texts).astype(np.float32).toarray()
    decisions = model.decision_function(feats)
    ml_preds = (decisions >= 0).astype(np.int8)

    final_preds = []
    for idx, text in enumerate(raw_texts):
        cleaned = clean_text(text, all_stopwords, tagalog_lemma, eng_lemma)
        rule_labels = rule_based_labels(cleaned, lexicons, label_order)
        combined = []
        for j in range(len(label_order)):
            if abs(decisions[idx, j]) < thr:
                combined.append(rule_labels[j])
            else:
                combined.append(int(ml_preds[idx, j]))
        final_preds.append(combined)
    return np.array(final_preds, dtype=np.int8)


def evaluate(y_true, y_pred, label_names):
    """Evaluate model performance."""
    print("=== Classification Report ===")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            zero_division=0,
        )
    )
    print("Hamming Loss:", hamming_loss(y_true, y_pred))
    print("Subset Accuracy:", accuracy_score(y_true, y_pred))


def count_rule_usage(raw_texts, model, vectorizer, label_order, thr=0.2):
    """Count percentage of rule usage per label."""
    feats = vectorizer.transform(raw_texts).astype(np.float32).toarray()
    decisions = model.decision_function(feats)
    rule_used = (np.abs(decisions) < thr).astype(np.int8)
    percent_used = rule_used.mean(axis=0) * 100
    return dict(zip(label_order, percent_used))


def main():
    """Main execution function."""
    # Setup
    setup_nltk()
    all_stopwords = load_stopwords()
    eng_lemma = WordNetLemmatizer()
    tagalog_lemma = load_lemma_dict('Multilabel-Hatespeech/filipino_lemma.txt')

    # Load and preprocess data
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    df = df.drop(columns=['Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'], axis=1)
    
    df["clean_text"] = df[TEXT_COLUMN].apply(
        lambda x: clean_text(x, all_stopwords, tagalog_lemma, eng_lemma)
    )
    df["tokens"] = df["clean_text"].apply(tokenize)
    df["token_text"] = df["tokens"].apply(lambda toks: " ".join(toks))
    
    # Split data (60/30/10)
    X_temp, X_val_original, y_temp, y_val = train_test_split(
        df["token_text"], df[LABEL_COLUMNS], test_size=0.10, random_state=42
    )
    
    X_train_original, X_test_original, y_train, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3333, random_state=42
    )
    
    # Vectorization
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        max_features=MAX_FEATS,
        ngram_range=NGRAM_RANGE,
        lowercase=True,
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_original).astype(np.float32).toarray()
    X_test = vectorizer.transform(X_test_original).astype(np.float32).toarray()
    X_val = vectorizer.transform(X_val_original).astype(np.float32).toarray()
    
    # Normalize features
    X_train = normalize(X_train_tfidf, norm='l2')
    X_test = normalize(X_test, norm='l2')
    X_val = normalize(X_val, norm='l2')
    
    # Train SVM model
    sgd = OneVsRestSGD(
        n_labels=len(LABEL_COLUMNS),
        n_features=X_train_tfidf.shape[1],
        C=1.0,
        lr=0.01,
        epochs=5,
    )
    sgd.fit(X_train, y_train.values.astype(np.int8))
    
    # Combine test and validation sets for evaluation
    X_combined_original = pd.concat([X_test_original, X_val_original])
    y_combined = pd.concat([y_test, y_val])
    X_combined_tfidf = vectorizer.transform(X_combined_original).astype(np.float32).toarray()
    X_combined = normalize(X_combined_tfidf, norm='l2')
    
    # Hybrid prediction
    y_combined_pred = hybrid_predict(
        X_combined_original.tolist(), 
        vectorizer, 
        sgd, 
        RULE_LEXICONS, 
        LABEL_COLUMNS,
        all_stopwords,
        tagalog_lemma,
        eng_lemma
    )
    
    print("SGD-RULE BASED HYBRID PERFORMANCE")
    evaluate(y_combined.values, y_combined_pred, LABEL_COLUMNS)
    
    # Pure model-only prediction
    print("\nPure SGD model performance:")
    y_pred = sgd.predict(X_combined)
    evaluate(y_combined.values, y_pred, LABEL_COLUMNS)
    
    # Rule usage analysis
    print("\nRule override % per label:")
    rule_usage = count_rule_usage(X_combined_original.tolist(), sgd, vectorizer, LABEL_COLUMNS)
    print(rule_usage)
    # Save the model
    with open('sgd_model.pkl', 'wb') as model_file:
        pickle.dump({
            'sgd_model': sgd,
            'vectorizer': vectorizer,
            'rule_lexicons': RULE_LEXICONS,
            'label_columns': LABEL_COLUMNS,
            'all_stopwords': all_stopwords,
            'tagalog_lemma': tagalog_lemma,
            'eng_lemma': eng_lemma
        }, model_file)


if __name__ == "__main__":
    main()