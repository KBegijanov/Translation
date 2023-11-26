import streamlit as st
#импортируем библиотеку streamlit, чтобы запустить код в приложении
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

#model_name = "facebook/bart-large-cnn"

#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

st.title("AI-content agregator")
#Название, которые будет видно у нас в приложении
text_input = st.text_area("Введите текст для конспектирования:", value="", height=200)
#Место для ввода текста, для дальнейшего конспектирования

if st.button("Конспектировать"):
    tokenized_text = summarizer(text_input, return_tensors="pt")
    #При нажатии на кнопку "перeвести" tokenizer принимает в качестве аргумента введенный текст
    agregation = model.generate(**tokenized_text)
    agregated_text = tokenizer.batch_decode(agregation, skip_special_tokens=True)
    st.write(agregated_text[0])
    #Выводится перевод
#rint(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
