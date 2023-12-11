import streamlit as st
import speech_recognition as sr
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pyttsx3
from langdetect import detect


# Load the BERT models and tokenizers for English and Chinese
@st.cache(allow_output_mutation=True)
def load_bert_models():
    tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
    model_en = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    tokenizer_cn = BertTokenizer.from_pretrained('bert-base-chinese')
    model_cn = BertForSequenceClassification.from_pretrained('bert-base-chinese')

    return tokenizer_en, model_en, tokenizer_cn, model_cn

# Convert audio to text using speech recognition
def convert_audio_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("Say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='zh-TW')  # For Chinese
        return text, 'zh'
    except sr.UnknownValueError:
        try:
            text = recognizer.recognize_google(audio)  # For English or other languages
            return text, 'en'
       except sr.UnknownValueError:
            return "Speech recognition could not understand audio", ''

# Preprocess text for sentiment analysis based on language
def preprocess_text(text, tokenizer, language):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# Analyze sentiment using the appropriate BERT model based on language
def analyze_sentiment_bert(text, tokenizer_en, model_en, tokenizer_cn, model_cn, language):
    if language == 'zh':
        tokenizer = tokenizer_cn
        model = model_cn
    else:
        tokenizer = tokenizer_en
        model = model_en

    inputs = preprocess_text(text, tokenizer, language)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    if predicted_class == 0:
        return "Seems negative." if language == 'en' else "负面情绪。"
    elif predicted_class == 1:
        return "Seems positive." if language == 'en' else "正面情绪。"

# Speak the analyzed sentiment
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    st.title("Audio Sentiment Analysis")

    # Load the BERT models and tokenizers
    tokenizer_en, model_en, tokenizer_cn, model_cn = load_bert_models()

    if st.button("Start Recording"):
        # Convert audio to text
        audio_text, detected_lang = convert_audio_to_text()

        if audio_text:
            st.write(f"Recognized text: {audio_text}")

            # Analyze sentiment using the appropriate BERT model
            sentiment = analyze_sentiment_bert(audio_text, tokenizer_en, model_en, tokenizer_cn, model_cn, detected_lang)
            st.write(f"Sentiment: {sentiment}")

            # Speak the sentiment
            speak_text(sentiment)

if __name__ == "__main__":
    main()
