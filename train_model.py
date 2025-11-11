import pandas as pd
from app.model import train_model, save_encoder
from app.utils import load_data
from app.feature_extraction import encode_text, encode_metadata, combine_features

features, labels = load_data()

text_features = encode_text(features['message'].tolist())
meta_features, encoder = encode_metadata(features)
X = combine_features(text_features, meta_features)
y = labels.values

model = train_model(X, y, epochs=5)

save_encoder(encoder)
