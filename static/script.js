document.getElementById('emailForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData();
  const text = document.getElementById('email_text').value.trim();
  const file = document.getElementById('file').files[0];
  if(text) fd.append('email_text', text);
  else if(file) fd.append('file', file);
  else { alert('Cole um texto ou envie um arquivo.'); return; }

  const btn = form.querySelector('button');
  btn.disabled = true; btn.textContent = 'Processando...';

  try {
    const resp = await fetch('/api/classify', { method: 'POST', body: fd });
    const data = await resp.json();
    if(resp.ok){
      document.getElementById('result').classList.remove('hidden');
      document.getElementById('category').textContent = data.category;
      document.getElementById('reply').textContent = data.suggested_reply;
      document.getElementById('extracted').textContent = data.extracted_text;
    } else {
      alert(data.error || 'Erro desconhecido');
    }
  } catch (err) {
    alert('Erro no servidor: ' + err.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Classificar e Gerar Resposta';
  }
});


const fileInput = document.getElementById('file');
const fileName = document.getElementById('file-name');

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        fileName.textContent = fileInput.files[0].name;
    } else {
        fileName.textContent = 'Nenhum arquivo escolhido';
    }
});