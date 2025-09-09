# train.py
import os
import joblib
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    text = text.lower()  # minusculas
    text = re.sub(r'\d+', '', text)  # remove numeros
    text = re.sub(r'[^\w\s]', '', text)  # remove pontuacao
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('portuguese'))
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

DATA = [

    # ---------------- PRODUTIVO ----------------
    ("Preciso que preparem a ata da reunião de ontem e enviem para a equipe até amanhã.", "Produtivo"),
    ("Revisar o contrato com o cliente e destacar pontos críticos antes da assinatura.", "Produtivo"),
    ("Organizar o cronograma do projeto X e compartilhar com o time até sexta-feira.", "Produtivo"),
    ("Responder dúvidas técnicas do cliente sobre o sistema de relatórios.", "Produtivo"),
    ("Finalizar a documentação da API e publicar no repositório oficial.", "Produtivo"),
    ("Gerar planilhas de acompanhamento de vendas para diretoria.", "Produtivo"),
    ("Elaborar contrato para novo cliente.", "Produtivo"),
    ("Agendar call de suporte para esclarecer dúvidas do cliente.", "Produtivo"),
    ("Atualizar o status das tarefas no Trello.", "Produtivo"),
    ("Escrever relatório de desempenho da campanha de marketing.", "Produtivo"),
    ("Atualizar o status das tarefas no Trello.", "Produtivo"),
    ("Marcar tarefas como concluídas no Trello.", "Produtivo"),
    ("Organizar cards no Trello e atualizar prioridades.", "Produtivo"),
    ("Adicionar comentários em tarefas no Trello.", "Produtivo"),
    ("Atualizar o status das tarefas no Trello.", "Produtivo"),
    ("Marcar tarefas como concluídas no Trello.", "Produtivo"),
    ("Organizar cards no Trello e atualizar prioridades.", "Produtivo"),
    ("Adicionar comentários em tarefas no Trello.", "Produtivo"),
    ("Criar novas tarefas no Jira para equipe.", "Produtivo"),
    ("Atualizar planilha de acompanhamento financeiro.", "Produtivo"),
    ("Enviar relatório de métricas do site para diretoria.", "Produtivo"),
    
    # ---------------- IMPRODUTIVO ----------------
    ("Ola vamos sair", "Improdutivo"),
    ("Oi gente, bom dia!", "Improdutivo"),
    ("Vamos almoçar fora hoje?", "Improdutivo"),
    ("E aí, vamos jogar mais tarde?", "Improdutivo"),
    ("Boa noite, até amanhã!", "Improdutivo"),
    ("Saudade de vocês, bora marcar algo", "Improdutivo"),
    ("Rolar Instagram por 1 hora durante o expediente.", "Improdutivo"),
    ("Enviar stickers engraçados no grupo da empresa.", "Improdutivo"),
    ("Comentar sobre o jogo de ontem durante a manhã toda.", "Improdutivo"),
    ("Postar piadas no Slack da empresa.", "Improdutivo"),

    # ---------------- PRODUTIVO ----------------
    ("Preciso que verifiquem o erro no sistema X. Quando puderem, me retornem com o passo a passo.", "Produtivo"),
    ("Temos um bug no login: o usuário não consegue resetar a senha.", "Produtivo"),
    ("Cliente solicita relatório atualizado do mês anterior. Favor enviar até sexta.", "Produtivo"),
    ("Estudar React 2h para projeto pessoal.", "Produtivo"),
    ("Finalizar relatório financeiro antes do prazo.", "Produtivo"),
    ("Responder e-mails pendentes da equipe.", "Produtivo"),
    ("Agendar reunião com cliente para alinhamento de projeto.", "Produtivo"),
    ("Treinar vendas com equipe interna durante 1h.", "Produtivo"),
    ("Preparar material de apresentação para reunião.", "Produtivo"),
    ("Revisar documentação técnica e corrigir erros.", "Produtivo"),
    ("Gerar planilhas de acompanhamento de vendas para diretoria.", "Produtivo"),
    ("Elaborar contrato para novo cliente.", "Produtivo"),
    ("Enviar orçamento atualizado para o cliente A.", "Produtivo"),
    ("Agendar call de suporte para esclarecer dúvidas do cliente.", "Produtivo"),
    ("Atualizar o status das tarefas no Trello.", "Produtivo"),
    ("Escrever relatório de desempenho da campanha de marketing.", "Produtivo"),
    ("Revisar código e aprovar PR no GitHub.", "Produtivo"),
    ("Configurar ambiente de testes no servidor.", "Produtivo"),
    ("Organizar arquivos no Google Drive para a equipe.", "Produtivo"),
    ("Fazer backup semanal do banco de dados.", "Produtivo"),
    ("Responder dúvidas técnicas de clientes via e-mail.", "Produtivo"),
    ("Preparar lista de materiais para treinamento de novos colaboradores.", "Produtivo"),
    ("Analisar métricas de acesso do site e criar relatório.", "Produtivo"),
    ("Definir cronograma do projeto e compartilhar com stakeholders.", "Produtivo"),
    ("Agendar feedback 1:1 com membro da equipe.", "Produtivo"),

    # ---------------- IMPRODUTIVO ----------------
    ("Parabéns pelo ótimo trabalho, obrigado!", "Improdutivo"),
    ("Feliz aniversário! Muitos sucessos!", "Improdutivo"),
    ("Obrigado pelo suporte ontem, excelente atendimento.", "Improdutivo"),
    ("Assistir série por 2h sem foco.", "Improdutivo"),
    ("Rolar redes sociais por tempo excessivo.", "Improdutivo"),
    ("Conversar sobre assuntos pessoais sem relação com trabalho.", "Improdutivo"),
    ("Jogar online sem objetivo definido.", "Improdutivo"),
    ("Ficar navegando na internet sem meta.", "Improdutivo"),
    ("Enviar memes para colegas de trabalho.", "Improdutivo"),
    ("Parabenizar alguém sem contexto de trabalho.", "Improdutivo"),
    ("Comentar sobre o jogo de futebol da noite passada.", "Improdutivo"),
    ("Discutir política em grupo de trabalho.", "Improdutivo"),
    ("Compartilhar vídeos engraçados no horário do expediente.", "Improdutivo"),
    ("Fazer piadas sem relação com o trabalho.", "Improdutivo"),
    ("Procrastinar a entrega de tarefas navegando em sites aleatórios.", "Improdutivo"),
    ("Passar tempo excessivo em aplicativos de mensagens.", "Improdutivo"),
    ("Assistir vídeos de entretenimento durante o expediente.", "Improdutivo"),
    ("Reclamar de assuntos pessoais em e-mails profissionais.", "Improdutivo"),
    ("Conversar sobre hobbies em vez de focar nas tarefas.", "Improdutivo"),
    ("Enviar correntes motivacionais no grupo da equipe.", "Improdutivo"),
    ("Compartilhar memes sobre chefes no horário de trabalho.", "Improdutivo"),
    ("Assistir transmissões de lives que não têm relação com trabalho.", "Improdutivo"),
    ("Discutir novela durante reunião de equipe.", "Improdutivo"),
    ("Ficar atualizando redes sociais sem motivo.", "Improdutivo"),
    ("Postar piadas no Slack da empresa.", "Improdutivo"),
    # Improdutivo
    ("Ola vamos sair", "Improdutivo"),
    ("Oi gente, bom dia!", "Improdutivo"),
    ("Vamos almoçar fora hoje?", "Improdutivo"),
    ("E aí, vamos jogar mais tarde?", "Improdutivo"),
    ("Boa noite, até amanhã!", "Improdutivo"),
    ("Saudade de vocês, bora marcar algo", "Improdutivo"),

]


# DataFrame
df = pd.DataFrame(DATA, columns=['text', 'label'])
df['text_p'] = df['text'].apply(preprocess_text)

X = df['text_p']
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluation
preds = pipeline.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, preds))

# Cross-validation score
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print("CV Accuracy scores:", cv_scores)
print("CV Mean Accuracy:", cv_scores.mean())

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(pipeline, 'models/model.pkl')
print("Modelo salvo em models/model.pkl")
