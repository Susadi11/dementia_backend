from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
import joblib
import whisper
import tempfile
import os


app = FastAPI(
    title="AI Dementia Detection API",
    description="Late Fusion Dementia Prediction using Audio + Text",
    version="1.0"
)



# Load ML models
print("Loading models...")

audio_model = joblib.load("models/best_audio_model.pkl")
audio_scaler = joblib.load("models/audio_scaler.pkl")

text_model = joblib.load("models/best_text_model.pkl")
text_tfidf = joblib.load("models/text_tfidf.pkl")
text_pca = joblib.load("models/text_pca.pkl")

""" label_encoder = joblib.load("models/label_encoder.pkl") """

whisper_model = whisper.load_model("base")

print("Models loaded successfully.")


# Audio feature extraction
def extract_audio_features(path, sr_target=16000):
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

    features = np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc_delta.mean(axis=1),
        mfcc_delta2.mean(axis=1),
        chroma.mean(axis=1),
        contrast.mean(axis=1),
        tonnetz.mean(axis=1),
    ])

    return features.reshape(1, -1)


# Whisper transcription

def transcribe_audio(path):
    result = whisper_model.transcribe(path)
    return result["text"].lower()


# Late fusion prediction
def late_fusion_predict(audio_feat, transcript, w_audio=0.6, w_text=0.4):
    #Audio probability
    audio_feat = audio_scaler.transform(audio_feat)
    audio_prob = audio_model.predict_proba(audio_feat)[0][1]

    #Text probability 
    tfidf_vec = text_tfidf.transform([transcript]).toarray()
    text_vec = text_pca.transform(tfidf_vec)
    text_prob = text_model.predict_proba(text_vec)[0][1]

    #Fusion
    fused_prob = w_audio * audio_prob + w_text * text_prob
    label = "Dementia" if fused_prob >= 0.5 else "Control"

    return {
        "audio_probability": float(audio_prob),
        "text_probability": float(text_prob),
        "fused_probability": float(fused_prob),
        "prediction": label
    }


#endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Dementia Detection API. Use the /predict endpoint to upload a speech recording for dementia prediction."}
    
@app.post("/predict")
async def predict_dementia(file: UploadFile = File(...)):
    """
    Upload a speech recording (.wav or .mp3)
    Returns dementia prediction using late fusion
    """

    # Save temp audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    try:
        # Extract audio features
        audio_feat = extract_audio_features(temp_path)

        # Transcribe speech
        transcript = transcribe_audio(temp_path)

        # Predict using late fusion
        result = late_fusion_predict(audio_feat, transcript)

        return {
            "transcript": transcript,
            **result
        }

    finally:
        os.remove(temp_path)
