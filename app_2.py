import streamlit as st
#импортируем библиотеку streamlit, чтобы запустить код в приложении
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#импортируем модель

model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

st.title("AI-переводчик с использованием Hugging Face и Streamlit")
#Название, которые будет видно у нас в приложении
text_input = st.text_area("Введите текст для перевода:", value="", height=200)
#Место для ввода текста, для дальнейшего перевода

if st.button("Перевести"):
    tokenized_text = tokenizer(text_input, return_tensors="pt")
    #При нажатии на кнопку "перeвести" tokenizer принимает в качестве аргумента введенный текст
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)
    st.write(translated_text[0])
    #Выводится перевод
