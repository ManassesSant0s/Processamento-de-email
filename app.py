# app.py
import os
import sys
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from utils import extract_text_from_pdf, preprocess_text
from datetime import datetime

# ==============================
# Configuração de debug
# ==============================
DEBUG = os.getenv("DEBUG_EMAIL_CLASSIFIER", "1") == "1"

def dprint(*args, **kwargs):
    """Print controlado por DEBUG com timestamp e flush imediato."""
    if DEBUG:
        print(f"[DEBUG {datetime.now().strftime('%H:%M:%S')}]", *args, **kwargs)
        sys.stdout.flush()

# ==============================
# OpenAI (opcional)
# ==============================
import openai
OPENAI_API_KEY = ("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")

print (OPENAI_API_KEY,"A CHAVE AQUIIIIIIIII")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def generate_reply_with_openai(category, email_text):
    if not OPENAI_API_KEY:
        return "Olá, obrigado pelo contato. Retornaremos em breve."
    
    prompt = (
        f"Você é um assistente que gera respostas formais curtas para emails em português.\n"
        f"Categoria detectada: {category}.\n"
        f"Email:\n{email_text}\n\n"
        f"Gere uma resposta profissional de 4-7 frases adequada à categoria."
    )
    
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content":prompt}],
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        dprint("Erro ao gerar resposta com OpenAI:", e)
        return "Olá, obrigado pelo contato. Retornaremos em breve."



# ==============================
# Carregamento do modelo
# ==============================
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        dprint("✅ Modelo carregado:", MODEL_PATH, "| tipo:", type(model))
        vec = getattr(model.named_steps, "tfidf", None)
        clf = getattr(model.named_steps, "clf", None)
        if vec:
            vocab_size = len(getattr(vec, "vocabulary_", {}))
            dprint(f"Vectorizer: ngram_range={getattr(vec, 'ngram_range', None)}, "
                   f"max_features={getattr(vec, 'max_features', None)}, vocab_size={vocab_size}")

          
    except Exception as e:
        print("❌ Erro ao carregar modelo:", e)
        sys.stdout.flush()
else:
    print("❌ Arquivo do modelo não encontrado:", MODEL_PATH)
    sys.stdout.flush()

# ==============================
# Explicabilidade
# ==============================
def _get_vectorizer_and_clf(pipeline):
    return getattr(pipeline.named_steps, "tfidf", None), getattr(pipeline.named_steps, "clf", None)

def explain_logreg_prediction(pipeline, raw_text, max_features=15):
    result = {"ok": False, "error": None}
    try:
        pre = preprocess_text(raw_text)
        vec, clf = _get_vectorizer_and_clf(pipeline)
        if not vec or not clf:
            result["error"] = "Pipeline não possui 'tfidf' e/ou 'clf'."
            return result

        X = vec.transform([pre])
        pred_class = clf.predict(X)[0]
        classes = list(getattr(clf, "classes_", []))
        probs = dict(zip(classes, clf.predict_proba(X)[0])) if hasattr(clf, "predict_proba") else None

        vocab = getattr(vec, "vocabulary_", {}) or {}
        feature_names = np.empty(len(vocab), dtype=object)
        for term, idx in vocab.items():
            feature_names[idx] = term

        class_idx = classes.index(pred_class)
        coef = getattr(clf, "coef_", None)
        contrib = []

        if coef is not None and coef.shape[0] >= 1:
            nz = X.nonzero()[1]
            for j in nz:
                value = X[0, j]
                weight = coef[class_idx, j]
                score = float(value * weight)
                contrib.append((feature_names[j], float(value), float(weight), score))
            contrib_sorted = sorted(contrib, key=lambda x: abs(x[3]), reverse=True)[:max_features]
        else:
            contrib_sorted = []

        result.update({
            "ok": True,
            "preprocessed": pre,
            "pred_class": pred_class,
            "classes": classes,
            "probs": probs,
            "top_contributions": [
                {"term": t, "tfidf_value": v, "weight_for_pred_class": w, "impact": sc}
                for (t, v, w, sc) in contrib_sorted
            ]
        })
        return result
    except Exception as e:
        result["error"] = f"Falha ao explicar predição: {e}"
        return result

# ==============================
# Flask app
# ==============================
app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/classify", methods=["POST"])
def classify():
    # Captura de texto
    text = ""
    if "email_text" in request.form and request.form["email_text"].strip():
        text = request.form["email_text"]
       
    elif "file" in request.files:
        f = request.files["file"]
        filename = (f.filename or "").lower()
        if filename.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(f.stream)
        else:
            return jsonify({"success": False, "error": "Formato não suportado"}), 400
    else:
        return jsonify({"success": False, "error": "Nenhum texto ou arquivo enviado"}), 400

    pre = preprocess_text(text)

    if model is None:
        return jsonify({"success": False, "error": "Modelo não carregado"}), 500

    try:
        category = model.predict([pre])[0]
    except Exception as e:
        return jsonify({"success": False, "error": f"Erro na predição: {e}"}), 500

    debug_info = explain_logreg_prediction(model, text)


    # Resposta sugerida (OpenAI ou fallback)
    suggested = generate_reply_with_openai(category, text)

    return jsonify({
        "success": True,
        "category": category,
        "suggested_reply": suggested,
        "extracted_text": text[:4000],
        "debug": debug_info
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
