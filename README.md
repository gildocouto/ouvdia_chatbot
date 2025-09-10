# 💡 OuvdIA Chatbot – Assistente Inteligente da Ouvidoria da RFB

Este aplicativo implementa uma interface de **chat interativo** utilizando **Gradio** e **vLLM**, com o modelo `meta-llama/Llama-3-8B-Instruct`.
O sistema foi desenvolvido para apoiar a **Ouvidoria da Receita Federal do Brasil**, garantindo clareza, precisão jurídica, conformidade com a LGPD e uso ético da Inteligência Artificial.

---

## ⚙️ Funcionalidades Principais

* **Modelo de linguagem em fluxo contínuo (streaming)** utilizando `vLLM`.
* **Interface de usuário em Gradio**, com:

  * Caixa de texto para instruções (Prompt).
  * Caixa de texto principal para entrada do usuário.
  * Campo opcional para **contexto adicional**.
  * Chatbot com **avatares personalizados**.
* **Feedback do usuário**: botões de 👍/👎 com registro em log.
* **Autenticação via `.env`** com lista de usuários válidos.
* **Registro de logs** detalhado em `/ouvdia/root/logs`.
* **Configuração de amostragem** personalizável (`temperature`, `top_p`, `max_tokens`).
* **Uso ético e seguro**: respeita LGPD, privacidade, transparência e acessibilidade.

---

## 📂 Estrutura do Projeto

* `ouvdia_chatbot_2.py` → Código principal do app.
* `/ouvdia/root/assets/` → Recursos estáticos (logos, imagens de avatar).
* `/ouvdia/root/logs/` → Diretório onde os logs são salvos automaticamente.
* `/ouvdia/root/USUARIOS_VALIDOS.env` → Arquivo com usuários autorizados no formato:

  ```
  USUARIOS_VALIDOS=usuario1:senha1,usuario2:senha2
  ```

---

## 🚀 Como Executar

### 1) Pré-requisitos

* Python 3.10+
* GPU compatível (ex.: NVIDIA V100)
* Pacotes necessários:

  ```bash
  pip install vllm gradio python-dotenv transformers
  ```

### 2) Configurar variáveis de ambiente

Crie o arquivo `/ouvdia/root/USUARIOS_VALIDOS.env` com usuários autorizados:

```bash
USUARIOS_VALIDOS=admin:1234,usuario:senha
```

### 3) Executar o servidor

```bash
python ouvdia_chatbot_2.py
```

O app ficará disponível em:
👉 [http://0.0.0.0:7860](http://0.0.0.0:7860)

---

## 📝 Exemplo de Uso

1. Preencha o campo **Prompt** com instruções (ex.: “Responda de forma resumida em tópicos”).
2. Digite o **Texto Principal** com sua solicitação.
3. (Opcional) Insira um **Contexto** adicional.
4. Acompanhe a resposta do modelo em **tempo real** no histórico do chat.
5. Utilize 👍 ou 👎 para avaliar a resposta e gerar feedback.

---

## 🔒 Segurança

* Autenticação obrigatória via `.env`.
* Registro de interações e feedback em log.
* Prompt inicial reforça **ética, transparência e conformidade legal**.
* Compatível com princípios da **LGPD**.

---

## 📊 Comparação: `ouvdia_chatbot_2.py` x uso direto de `vLLM`

| Item                       | `ouvdia_chatbot_2.py`                     | `vLLM` puro                          |
| -------------------------- | ----------------------------------------- | ------------------------------------ |
| Interface de usuário       | ✅ Gradio com chat, avatares e feedback    | ❌ Necessário implementar manualmente |
| Streaming de respostas     | ✅ Sim                                     | ✅ Sim                                |
| Autenticação               | ✅ Via `.env`                              | ❌ Não incluída                       |
| Registro de logs           | ✅ Estruturado em `/ouvdia/root/logs`      | ❌ Não incluso                        |
| Uso ético/LGPD             | ✅ Prompt inicial orientado à conformidade | ❌ Responsabilidade do usuário        |
| Configuração pronta p/ RFB | ✅ Logos, mensagens institucionais         | ❌ Genérico                           |

---

## 📄 Licença

Uso restrito à **Receita Federal do Brasil**, conforme políticas institucionais de privacidade, proteção de dados e ética em IA.
Baseado em bibliotecas de código aberto sob as respectivas licenças (vLLM, Gradio, Hugging Face Transformers).
