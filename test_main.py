from fastapi.testclient import TestClient
from main import app

client = TestClient (app)

#проверяем работоспособность
def test_read_main():
    response = client.get ("/")
    assert response.status_code == 200   #статус = 200, значит сайт работает
    assert response.json() == {"message": "Hello! It's the translator."}  #на главной странице должно быть Hello! It's the translator, так как прописывали это в коде документа


def test_translation_ru():
    response = client.post ("/translate/", 
                            json = {"input_text": "Здравствуйте, как ваши дела?"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data["translated_text"] == "Hello, how are you?"

def test_translation_en():
    response = client.post ("/translate/", 
                            json = {"input_text": "Hello, how are you?"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data["translated_text"] == "Привет, как дела?"