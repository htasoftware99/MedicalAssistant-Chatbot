import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Modeli ve tokenizer'ı yükleyin
model_path = 'Chatbot/Python/Models/Models_local/local_medical_assistant_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Başlık
st.title("Medical Assistant Chatbot")

# Kullanıcı giriş alanı
input_text = st.text_area("Ask your question:", "")

# Yanıt üretmek için bir düğme
if st.button("Reply"):
    if input_text.strip() != "":
        # Girdiyi tokenize edin
        inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=128)

        # Modeli değerlendirme moduna alın
        model.eval()

        # Yanıt üretin
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=512,       # Maksimum yanıt uzunluğu
                num_beams=2,          # Beam search
                early_stopping=True,  # Erken durdurma
                no_repeat_ngram_size=1 # Tekrarları önlemek için
            )

        # Yanıtı decode edin
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(f"Doctor AI: {generated_text}")
    else:
        st.warning("Please ask a question.")

