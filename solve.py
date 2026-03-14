import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def extract_handcrafted_features(texts: np.ndarray) -> np.ndarray:
    rows: list[list[Any]] = []
    for text in texts:
        words = text.split()
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

        char_len = len(text)
        word_count = len(words)
        num_sentences = max(len(sentences), 1)

        sent_word_lens = [len(s.split()) for s in sentences] if sentences else [0]
        avg_sent_len = np.mean(sent_word_lens)
        sent_len_var = np.var(sent_word_lens)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        unique_words = len(set(w.lower() for w in words))
        vocab_richness = unique_words / max(word_count, 1)

        upper_ratio = sum(1 for c in text if c.isupper()) / max(char_len, 1)
        allcaps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        allcaps_ratio = allcaps_words / max(word_count, 1)

        punct_rate = sum(1 for c in text if c in ".,!?;:") / max(char_len, 1)
        comma_rate = text.count(",") / max(char_len, 1)
        repeated_chars = len(re.findall(r"(.)\1{2,}", text)) / max(char_len, 1)
        newline_rate = text.count("\n") / max(char_len, 1)

        has_bullets = 1 if re.search(r"^\s*[-*]\s", text, re.MULTILINE) else 0
        has_numbered = 1 if re.search(r"^\s*\d+[.)]\s", text, re.MULTILINE) else 0

        is_short = 1 if char_len < 300 else 0
        is_single_sentence = 1 if num_sentences <= 1 else 0
        is_long = 1 if char_len > 2000 else 0

        tl = text.lower()
        contractions = sum(
            1
            for p in [
                "i'm",
                "it's",
                "don't",
                "can't",
                "won't",
                "i've",
                "you're",
                "that's",
                "isn't",
                "didn't",
                "couldn't",
                "they're",
                "we're",
            ]
            if p in tl
        )
        contraction_rate = contractions / max(word_count, 1)

        formal_transitions = sum(
            1
            for p in [
                "moreover,",
                "furthermore,",
                "in conclusion,",
                "in summary,",
                "additionally,",
                "therefore,",
                "however,",
            ]
            if p in tl
        )

        rows.append(
            [
                char_len,
                word_count,
                num_sentences,
                avg_sent_len,
                sent_len_var,
                avg_word_len,
                vocab_richness,
                upper_ratio,
                allcaps_ratio,
                punct_rate,
                comma_rate,
                repeated_chars,
                newline_rate,
                has_bullets,
                has_numbered,
                is_short,
                is_single_sentence,
                contraction_rate,
                formal_transitions,
                is_long,
            ]
        )

    return np.array(rows, dtype=np.float32)


def build_sparse_features(
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    word_ngram: tuple[int, int],
    char_ngram: tuple[int, int],
    word_max_features: int,
    char_max_features: int,
    word_min_df: int,
    char_min_df: int,
    include_hcf: bool,
) -> tuple[csr_matrix, csr_matrix]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        max_features=word_max_features,
        sublinear_tf=True,
        min_df=word_min_df,
        strip_accents="unicode",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        max_features=char_max_features,
        sublinear_tf=True,
        min_df=char_min_df,
    )

    x_word_train = csr_matrix(word_vectorizer.fit_transform(x_train_raw)) # type: ignore
    x_word_test = csr_matrix(word_vectorizer.transform(x_test_raw)) # type: ignore
    x_char_train = csr_matrix(char_vectorizer.fit_transform(x_train_raw)) # type: ignore
    x_char_test = csr_matrix(char_vectorizer.transform(x_test_raw)) # type: ignore

    train_parts = [x_word_train, x_char_train]
    test_parts = [x_word_test, x_char_test]

    if include_hcf:
        hcf_train = extract_handcrafted_features(x_train_raw)
        hcf_test = extract_handcrafted_features(x_test_raw)
        scaler = StandardScaler()
        hcf_train_scaled = scaler.fit_transform(hcf_train) # type: ignore
        hcf_test_scaled = scaler.transform(hcf_test)
        train_parts.append(csr_matrix(hcf_train_scaled))
        test_parts.append(csr_matrix(hcf_test_scaled))

    x_train = csr_matrix(hstack(train_parts)) # type: ignore
    x_test = csr_matrix(hstack(test_parts)) # type: ignore
    return x_train, x_test


def get_oof_and_test_proba(
    x_train: csr_matrix,
    y_train: np.ndarray,
    x_test: csr_matrix,
    model_params: dict[str, Any],
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y_train)
    n_classes = len(classes)

    oof_proba = np.zeros((x_train.shape[0], n_classes), dtype=np.float64)
    test_proba_folds = np.zeros((x_test.shape[0], n_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_idx, val_idx in skf.split(x_train, y_train):
        model = LogisticRegression(**model_params)
        model.fit(x_train[fit_idx], y_train[fit_idx])
        oof_proba[val_idx] = model.predict_proba(x_train[val_idx])
        test_proba_folds += model.predict_proba(x_test)

    test_proba = test_proba_folds / n_splits
    return oof_proba, test_proba


def main() -> None:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    x_train_raw = train["TEXT"].to_numpy()
    y_train = train["LABEL"].to_numpy()
    x_test_raw = test["TEXT"].to_numpy()
    test_ids = test.iloc[:, 0].to_numpy()

    classes = np.unique(y_train)

    # Model A
    x_train_a, x_test_a = build_sparse_features(
        x_train_raw,
        x_test_raw,
        word_ngram=(1, 3),
        char_ngram=(2, 5),
        word_max_features=50000,
        char_max_features=50000,
        word_min_df=2,
        char_min_df=3,
        include_hcf=True,
    )
    oof_a, test_a = get_oof_and_test_proba(
        x_train_a,
        y_train,
        x_test_a,
        {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 100.0,
            "max_iter": 1200,
        },
    )

    # Model B
    x_train_b, x_test_b = build_sparse_features(
        x_train_raw,
        x_test_raw,
        word_ngram=(1, 2),
        char_ngram=(2, 6),
        word_max_features=60000,
        char_max_features=80000,
        word_min_df=2,
        char_min_df=2,
        include_hcf=False,
    )
    oof_b, test_b = get_oof_and_test_proba(
        x_train_b,
        y_train,
        x_test_b,
        {
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 60.0,
            "max_iter": 1200,
        },
    )

    # Model C
    x_train_c, x_test_c = build_sparse_features(
        x_train_raw,
        x_test_raw,
        word_ngram=(1, 3),
        char_ngram=(3, 5),
        word_max_features=45000,
        char_max_features=70000,
        word_min_df=1,
        char_min_df=2,
        include_hcf=True,
    )
    oof_c, test_c = get_oof_and_test_proba(
        x_train_c,
        y_train,
        x_test_c,
        {
            "class_weight": None,
            "solver": "lbfgs",
            "C": 30.0,
            "max_iter": 1200,
        },
    )

    # Soft-vote baseline
    base_test_proba = 0.45 * test_a + 0.35 * test_b + 0.20 * test_c

    # Meta-learner stacking
    x_meta_train = np.hstack([oof_a, oof_b, oof_c])
    x_meta_test = np.hstack([test_a, test_b, test_c])

    meta_model = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        C=6.0,
        max_iter=2000,
    )
    meta_model.fit(x_meta_train, y_train)
    meta_test_proba = meta_model.predict_proba(x_meta_test)

    # Blend meta and base predictions
    test_proba = 0.75 * meta_test_proba + 0.25 * base_test_proba

    test_predictions = classes[np.argmax(test_proba, axis=1)]

    submission = pd.DataFrame({"ID": test_ids, "LABEL": test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()