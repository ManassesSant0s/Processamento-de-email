import joblib
from utils import preprocess_text

# Carrega o modelo treinado
model = joblib.load('models/model.pkl')

# Lista de testes (texto e rótulo esperado)
tests = [
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
]

# Rodando os testes
print("\n=== Testes de classificação automatizados ===\n")
acertos = 0
for text, expected in tests:
    pre = preprocess_text(text)
    predicted = model.predict([pre])[0]
    result = "✅" if predicted == expected else "❌"
    if result == "✅":
        acertos += 1
    print(f"Texto: {text}")
    print(f"Esperado: {expected} | Previsto: {predicted} {result}\n")

total = len(tests)
print(f"✅ Acertos: {acertos}/{total} ({acertos/total*100:.1f}%)")
print(f"❌ Erros: {total-acertos}/{total} ({(total-acertos)/total*100:.1f}%)\n")
