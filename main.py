from fastapi import FastAPI 
from transformers import AutoModelWithLMHead, AutoTokenizer 
from pydantic import BaseModel

app = FastAPI()

class TranslationRequest(BaseModel): 
    input_text: str

model_ru_en = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-ru-en") 
tokenizer_ru_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en") 
model_en_ru = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-ru") 
tokenizer_en_ru = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

def translate_text(text, model, tokenizer): 
    input_ids = tokenizer.encode(text, return_tensors="pt") 
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True) 
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True) 
    return translated_text

@app.post("/translate") 
def translate(request: TranslationRequest): 
    input_text = request.input_text 
    # Определяем язык текста 
    source_lang = "en" if all(ord(c) < 128 for c in input_text) else "ru"

    if source_lang == "en": 
        translated_text = translate_text(input_text, model_en_ru, tokenizer_en_ru) 
    else: 
        translated_text = translate_text(input_text, model_ru_en, tokenizer_ru_en)

    return {"input_text": input_text, "translated_text": translated_text}
    
