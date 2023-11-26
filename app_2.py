"""from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")

@app.get("/")
def root():
    return {"message": "Hello"}
    
@app.post("/predict/")
def predict(item: Item):
    return pipe(item.text )[0]"""

import streamlit as st
#импортируем библиотеку streamlit, чтобы запустить код в приложении
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    #Выводится перевод"""
