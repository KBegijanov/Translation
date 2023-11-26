from fastapi import FastAPI
from transformers import MT5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

# Загрузка предобученной модели mT5 и токенизатора
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')


@app.post("/translate/")
def translate_text(translation_request: dict):
    source_text = translation_request["text"]
    source_lang = translation_request["source_lang"]
    target_lang = translation_request["target_lang"]

    # Получение токенов и конвертация входного текста
    input_ids = tokenizer.encode(source_text, return_tensors="pt")
    translated_text = model.generate(input_ids=input_ids, decoder_start_token_id=model.config.pad_token_id)

    # Осуществление обратной конвертации и декодирование выходного текста
    translated_text = tokenizer.decode(translated_text[0], skip_special_tokens=True)

    return {"translated_text": translated_text}


@app.post("/translate/english-to-russian")
def translate_english_to_russian(translation_request: dict):
    translation_request["source_lang"] = "en"
    translation_request["target_lang"] = "ru"
    return translate_text(translation_request)


@app.post("/translate/russian-to-english")
def translate_russian_to_english(translation_request: dict):
    translation_request["source_lang"] = "ru"
    translation_request["target_lang"] = "en"
    return translate_text(translation_request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



"""from transformers import MarianMTModel, MarianTokenizer
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Загрузка модели и токенизатора для перевода текста
model_name = "Helsinki-NLP/opus-mt-en-ru"  # Модель для перевода с английского на русский
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)


# Модель запроса для FastAPI
class TranslationRequest(BaseModel):
    text: str  # Текст для перевода


# Маршрут для перевода текста
@app.post("/translate")
def translate_text(request: TranslationRequest):
    # Токенизация входного текста
    inputs = tokenizer.encode(request.text, return_tensors="pt")

    # Перевод текста с использованием модели
    translated = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {"translation": translated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) """

"""
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
print(tokenizer.decode(tokens[0], skip_special_tokens=True))"""

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
