from fastapi.testclient import TestClient
from main import app

client = TestClient (app)

#проверяем работоспособность
def test_read_main():
    response = client.get ("/")
    assert response.status_code == 200   #статус = 200, значит сайт работает
    assert response.json() == {"message": "Hello! It's the translator."}  #на главной странице должно быть Hello! It's the translator, так как прописывали это в коде документа


