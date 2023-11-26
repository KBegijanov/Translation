from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("summarization", model="facebook/bart-large-cnn")

@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text )[0]
