import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer

# Load text embedding model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------
# TEXT ENCODING
# ------------------------------
def encode_text(messages):
    if not messages:
        return np.zeros((0, text_model.get_sentence_embedding_dimension()))
    embeddings = text_model.encode(messages, show_progress_bar=False)
    return np.array(embeddings)

# ------------------------------
# METADATA ENCODING
# ------------------------------
def encode_metadata(df, encoder=None):
    cat_cols = ["pipeline_id", "stage_name", "job_name", "task_name", "branch", "user"]
    meta_df = df[cat_cols].astype(str)

    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(meta_df)
    else:
        encoded = encoder.transform(meta_df)

    return encoded, encoder

# ------------------------------
# FEATURE COMBINATION
# ------------------------------
def combine_features(text_features, meta_features):
    if not isinstance(text_features, np.ndarray):
        text_features = np.array(text_features)
    if not isinstance(meta_features, np.ndarray):
        meta_features = np.array(meta_features)

    n_samples = min(text_features.shape[0], meta_features.shape[0])
    text_features = text_features[:n_samples]
    meta_features = meta_features[:n_samples]

    text_features = np.atleast_2d(text_features)
    meta_features = np.atleast_2d(meta_features)

    return np.concatenate([text_features, meta_features], axis=1)
