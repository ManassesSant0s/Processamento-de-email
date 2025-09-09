# Email Classifier — Projeto AutoU

Aplicação simples que classifica emails em **Produtivo** ou **Improdutivo** e gera uma resposta automática sugerida.

## Como usar localmente

1. Clone o repositório:
   ```bash
   git clone <repo-url>
   cd email-ai-classifier
   ```

2. Crie e ative ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Treine o modelo (gera models/model.pkl):
   ```bash
   python train.py
   ```

5. Rode a aplicação:
   ```bash
   python app.py
   ```

6. Acesse: http://127.0.0.1:5000/

## Deploy
- Heroku / Render: apontar para este repositório. Build: `pip install -r requirements.txt`. Start: `gunicorn app:app`.

## Observações
- Melhore o dataset em `train.py` para aumentar acurácia.
- Para respostas mais naturais, integre uma API como OpenAI (snippet no README original).
