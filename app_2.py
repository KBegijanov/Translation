import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-3b-v2")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-base-alpha-3b-v2",
  trust_remote_code=True,
  torch_dtype="auto",
)
model.cuda()
inputs = tokenizer("The weather is always wonderful", return_tensors="pt").to("cuda")
tokens = model.generate(
  **inputs,
  max_new_tokens=64,
  temperature=0.75,
  top_p=0.95,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))

"""from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Item(BaseModel):
    text: str

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

@app.get("/")
def root():
    return {"message": "Hello"}
    
@app.post("/predict/")
def predict(item: Item):
    return tokenizer(item.text )[0]
"""
"""import streamlit as st
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
