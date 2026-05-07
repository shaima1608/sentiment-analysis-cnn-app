
import os
import json
import pickle
import pandas as pd
import streamlit as st

from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Dropout,
    Dense,
    BatchNormalization
)
from keras.preprocessing.sequence import pad_sequences

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CNN Sentiment Analysis App",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "."

MODEL_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "improved_cnn.weights.h5")
TOKENIZER_PATH = os.path.join(PROJECT_DIR, "tokenizer.pkl")
METADATA_PATH = os.path.join(PROJECT_DIR, "metadata.json")
IMDB_COMPARISON_PATH = os.path.join(PROJECT_DIR, "imdb_model_comparison.csv")
TWITTER_COMPARISON_PATH = os.path.join(PROJECT_DIR, "twitter_model_comparison.csv")

# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = {
    "Improved CNN weights": MODEL_WEIGHTS_PATH,
    "Tokenizer": TOKENIZER_PATH,
    "Metadata": METADATA_PATH,
    "IMDB comparison": IMDB_COMPARISON_PATH,
    "Twitter comparison": TWITTER_COMPARISON_PATH
}

missing_files = []

for name, path in required_files.items():
    if not os.path.exists(path):
        missing_files.append(f"{name}: {path}")

if missing_files:
    st.error("Some required files are missing.")
    st.write("Run the training notebook first, then restart the Streamlit app.")
    st.code("\n".join(missing_files))
    st.stop()

# ============================================================
# LOAD RESOURCES
# ============================================================
def build_improved_cnn(vocab_size, embedding_dim, max_sequence_length):
    sequence_input = Input(shape=(max_sequence_length,), dtype="int32")

    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_sequence_length,
        trainable=True
    )(sequence_input)

    x = Conv1D(256, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 4, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPooling1D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(sequence_input, output)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


@st.cache_resource
def load_cnn_model():
    with open(TOKENIZER_PATH, "rb") as f:
        saved_tokenizer = pickle.load(f)

    with open(METADATA_PATH, "r") as f:
        saved_metadata = json.load(f)

    vocab_size = len(saved_tokenizer.word_index) + 1
    embedding_dim = saved_metadata.get("embedding_dim", 300)
    max_sequence_length = saved_metadata.get("improved_max_sequence_length", 80)

    model = build_improved_cnn(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length
    )

    model.load_weights(MODEL_WEIGHTS_PATH)

    return model

@st.cache_resource
def load_saved_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metadata_file():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_csv_tables():
    imdb_df = pd.read_csv(IMDB_COMPARISON_PATH)
    twitter_df = pd.read_csv(TWITTER_COMPARISON_PATH)
    return imdb_df, twitter_df

try:
    model = load_cnn_model()
    tokenizer = load_saved_tokenizer()
    metadata = load_metadata_file()
    imdb_comparison, twitter_comparison = load_csv_tables()
except Exception as e:
    st.error("Error while loading model files.")
    st.code(str(e))
    st.stop()

MAX_SEQUENCE_LENGTH = metadata.get("improved_max_sequence_length", 80)

# ============================================================
# FUNCTIONS
# ============================================================

def spacy_tokenize(text):
    import re

    stop_words = {
        "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
        "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
        "as", "by", "at", "from", "this", "that", "these", "those", "it", "its"
    }

    words = re.findall(r"\b[a-zA-Z']+\b", str(text).lower())

    tokens = [
        word for word in words
        if word not in stop_words and word.strip() != ""
    ]

    return tokens
    
def preprocess_text(text):
    tokens = spacy_tokenize(text)
    cleaned_text = " ".join(tokens)

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    return padded_sequence, cleaned_text


def predict_sentiment(text):
    padded_sequence, cleaned_text = preprocess_text(text)

    prediction = model.predict(padded_sequence, verbose=0)[0]

    positive_confidence = float(prediction[0])
    negative_confidence = float(prediction[1])

    if positive_confidence >= negative_confidence:
        sentiment = "Positive"
        confidence = positive_confidence
    else:
        sentiment = "Negative"
        confidence = negative_confidence

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_confidence": positive_confidence,
        "negative_confidence": negative_confidence,
        "cleaned_text": cleaned_text
    }


def show_table_as_markdown(df):
    st.markdown(df.to_markdown(index=False))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🤖 CNN App")
    st.write("Use the arrow icon at the top-left to hide/show the sidebar.")

    st.divider()

    st.subheader("⚙️ Model Settings")
    st.write("**Model:** Improved CNN")
    st.write("**Embedding:**", metadata.get("embedding_choice", "Not available"))
    st.write("**Sequence Length:**", MAX_SEQUENCE_LENGTH)

    st.divider()

    st.subheader("🎬 IMDB Results")
    st.write("**Original Accuracy:**", round(metadata.get("original_imdb_accuracy", 0), 4))
    st.write("**Original Loss:**", round(metadata.get("original_imdb_loss", 0), 4))
    st.write("**Improved Accuracy:**", round(metadata.get("improved_imdb_accuracy", 0), 4))
    st.write("**Improved Loss:**", round(metadata.get("improved_imdb_loss", 0), 4))

    st.divider()

    st.subheader("🐦 Twitter Results")
    st.write("**Original Accuracy:**", round(metadata.get("original_twitter_accuracy", 0), 4))
    st.write("**Original Loss:**", round(metadata.get("original_twitter_loss", 0), 4))
    st.write("**Improved Accuracy:**", round(metadata.get("improved_twitter_accuracy", 0), 4))
    st.write("**Improved Loss:**", round(metadata.get("improved_twitter_loss", 0), 4))

    st.divider()
    st.success("App is ready ✅")

# ============================================================
# MAIN PAGE
# ============================================================

st.title("🎬 Sentiment Analysis Web App Using CNN")

st.write(
    """
    This web app predicts whether text sentiment is **Positive** or **Negative**
    using an improved CNN model.
    """
)

st.info("Use the arrow icon at the top-left of the page to hide or show the sidebar.")

st.subheader("📌 Quick Summary")

summary_table = pd.DataFrame(
    {
        "Item": ["Original CNN Accuracy", "Improved CNN Accuracy", "Sequence Length"],
        "Value": [
            round(metadata.get("original_imdb_accuracy", 0), 4),
            round(metadata.get("improved_imdb_accuracy", 0), 4),
            MAX_SEQUENCE_LENGTH
        ]
    }
)

show_table_as_markdown(summary_table)

st.divider()

# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Choose a page:",
    [
        "📝 Single Sentence",
        "📂 CSV Upload",
        "📊 Model Comparison",
        "📌 Explanation"
    ],
    horizontal=True
)

st.divider()

# ============================================================
# PAGE 1: SINGLE SENTENCE
# ============================================================

if page == "📝 Single Sentence":
    st.header("📝 Predict One Sentence")

    user_text = st.text_input(
        "Enter a sentence:",
        placeholder="Example: This movie was amazing and I really enjoyed it."
    )

    if st.button("🔍 Predict Sentiment", type="primary"):
        if user_text.strip() == "":
            st.warning("Please enter a sentence first.")
        else:
            result = predict_sentiment(user_text)

            st.subheader("Prediction Result")

            if result["sentiment"] == "Positive":
                st.success("✅ Sentiment: Positive")
            else:
                st.error("❌ Sentiment: Negative")

            result_table = pd.DataFrame(
                {
                    "Metric": [
                        "Confidence Score",
                        "Positive Confidence",
                        "Negative Confidence"
                    ],
                    "Value": [
                        round(result["confidence"], 4),
                        round(result["positive_confidence"], 4),
                        round(result["negative_confidence"], 4)
                    ]
                }
            )

            show_table_as_markdown(result_table)

            st.subheader("Cleaned Text Used by the Model")
            st.code(result["cleaned_text"])

# ============================================================
# PAGE 2: CSV UPLOAD
# ============================================================

elif page == "📂 CSV Upload":
    st.header("📂 Upload CSV for Multiple Predictions")

    st.write("Upload a CSV file that contains a text column.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            csv_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error("Could not read the CSV file.")
            st.code(str(e))
            st.stop()

        st.subheader("CSV Preview")
        show_table_as_markdown(csv_df.head())

        text_column = st.selectbox("Select the text column:", csv_df.columns)

        if st.button("🚀 Predict CSV Sentiments", type="primary"):
            results = []
            texts = csv_df[text_column].astype(str).tolist()

            for text in texts:
                result = predict_sentiment(text)

                results.append(
                    {
                        "Original Text": text,
                        "Cleaned Text": result["cleaned_text"],
                        "Predicted Sentiment": result["sentiment"],
                        "Confidence": round(result["confidence"], 4),
                        "Positive Confidence": round(result["positive_confidence"], 4),
                        "Negative Confidence": round(result["negative_confidence"], 4)
                    }
                )

            results_df = pd.DataFrame(results)

            st.success("CSV predictions completed.")

            st.subheader("Prediction Results")
            show_table_as_markdown(results_df.head(20))

            csv_output = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Full Predictions CSV",
                data=csv_output,
                file_name="sentiment_predictions.csv",
                mime="text/csv"
            )

# ============================================================
# PAGE 3: MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":
    st.header("📊 Original CNN vs Improved CNN")

    st.subheader("🎬 IMDB Dataset Comparison")
    show_table_as_markdown(imdb_comparison)

    st.subheader("🐦 Twitter Dataset Comparison")
    show_table_as_markdown(twitter_comparison)

    original_acc = metadata.get("original_imdb_accuracy", 0)
    improved_acc = metadata.get("improved_imdb_accuracy", 0)
    original_loss = metadata.get("original_imdb_loss", 0)
    improved_loss = metadata.get("improved_imdb_loss", 0)

    st.subheader("🏆 Which Model Is Better?")

    if improved_acc > original_acc:
        st.success("The improved CNN is better because it achieved higher IMDB test accuracy.")
    elif improved_acc < original_acc:
        st.warning("The original CNN is better because it achieved higher IMDB test accuracy.")
    else:
        if improved_loss < original_loss:
            st.success("Both models have the same accuracy, but the improved CNN has lower loss.")
        else:
            st.warning("Both models have the same accuracy, but the original CNN has lower loss.")

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Original CNN IMDB Accuracy",
                "Original CNN IMDB Loss",
                "Improved CNN IMDB Accuracy",
                "Improved CNN IMDB Loss",
                "Original CNN Twitter Accuracy",
                "Original CNN Twitter Loss",
                "Improved CNN Twitter Accuracy",
                "Improved CNN Twitter Loss"
            ],
            "Value": [
                round(metadata.get("original_imdb_accuracy", 0), 4),
                round(metadata.get("original_imdb_loss", 0), 4),
                round(metadata.get("improved_imdb_accuracy", 0), 4),
                round(metadata.get("improved_imdb_loss", 0), 4),
                round(metadata.get("original_twitter_accuracy", 0), 4),
                round(metadata.get("original_twitter_loss", 0), 4),
                round(metadata.get("improved_twitter_accuracy", 0), 4),
                round(metadata.get("improved_twitter_loss", 0), 4)
            ]
        }
    )

    st.subheader("Full Metrics Summary")
    show_table_as_markdown(summary_df)

# ============================================================
# PAGE 4: EXPLANATION
# ============================================================

elif page == "📌 Explanation":
    st.header("📌 Project Explanation")

    st.subheader("✅ Web App Requirements")
    st.write(
        """
        This app satisfies the requirements because it:
        
        - Takes a sentence as input.
        - Predicts sentiment as Positive or Negative.
        - Shows a confidence score.
        - Allows CSV upload for multiple texts.
        - Uses the CNN pipeline from the lab.
        """
    )

    st.subheader("🧠 Model Improvement")
    st.write(
        """
        The improved CNN includes meaningful changes:
        
        - Sequence length increased from 50 to 80.
        - The embedding layer is trainable.
        - Conv1D filters were changed.
        - BatchNormalization was added.
        - Dropout was increased.
        - GlobalMaxPooling1D was used instead of Flatten.
        - Softmax output was used for two-class classification.
        """
    )

    st.subheader("🐦 Another Dataset")
    st.write(
        """
        The model was tested on the Twitter sentiment dataset.
        Performance can change because IMDB reviews and tweets are different domains.
        IMDB reviews are longer and more formal, while tweets are shorter and may contain slang,
        hashtags, usernames, emojis, abbreviations, and sarcasm.
        """
    )
