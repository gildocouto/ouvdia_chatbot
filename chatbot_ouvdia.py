# chatbot_ouvdia.py
# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import gradio as gr
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter

# ========== Configuração do logging ==========

# Define o diretório de logs de forma mais direta
diretorio_logs = Path('/ouvdia/logs')
diretorio_logs.mkdir(parents=True, exist_ok=True)

# Nome do arquivo de log com data
data_atual = datetime.now().strftime("%Y-%m-%d")
caminho_arquivo_logs = diretorio_logs / f'ouvdia_logs_{data_atual}.log'

# Configuração do logging
logging.basicConfig(
    filename=caminho_arquivo_logs,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ========== Definição das classes ==========

class ModeloLinguagemContinuo:
    """
    Classe para inicializar e gerenciar um modelo de linguagem de fluxo contínuo
    usando vLLM.
    """

    def __init__(
        self,
        modelo: str,
        tipo_dados: str = "auto",
        quantizacao: Optional[str] = None,
        **kwargs,  # Captura args extras (max_model_len, gpu_memory_utilization, etc.)
    ) -> None:
        """
        Construtor da classe.
        
        Os **kwargs são desempacotados e passados diretamente para EngineArgs.
        """
        argumentos_motor = EngineArgs(
            model=modelo,
            quantization=quantizacao,
            dtype=tipo_dados,
            enforce_eager=False,
            device="cuda",
            **kwargs  # Aplica configurações como max_model_len
        )
        self.llm_motor = LLMEngine.from_engine_args(
            argumentos_motor,
            usage_context=UsageContext.LLM_CLASS
        )
        self.contador_requisicoes = Counter()

    def gerar(
        self,
        prompt: str,
        parametros_amostragem: Optional[SamplingParams] = None
    ) -> Iterator[RequestOutput]:
        """Gera a resposta do modelo de forma iterativa (stream)."""
        id_requisicao = str(next(self.contador_requisicoes))
        self.llm_motor.add_request(id_requisicao, prompt, parametros_amostragem)

        while self.llm_motor.has_unfinished_requests():
            saidas_etapa = self.llm_motor.step()
            for saida in saidas_etapa:
                yield saida


class InterfaceUsuario:
    """
    Classe que gerencia a interface do usuário via Gradio.
    """

    def __init__(
        self,
        llm: ModeloLinguagemContinuo,
        tokenizador: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        parametros_amostragem: Optional[SamplingParams] = None,
        system_prompt: str = (
            "Seu nome é OuvdIA, você é a Assistente Inteligente da Ouvidoria da Receita Federal do Brasil."
            " Você foi desenvolvida por um equipe multidisciplinar composta por servidores da área de negócios e especialistas em Inteligência Artifical."
            " Você fornece respostas concisas, precisas e bem estruturadas."
            " Nas suas respostas garanta clareza, exatidão jurídica e conformidade com a legislação vigente, especialmente a LGPD,"
            " bem como o uso ético e responsável de Inteligência Artifical."
            " Você utiliza princípios éticos, incluindo transparência, responsabilidade, justiça, privacidade e segurança."
            " Seu tom é profissional e respeitoso."
            " Use a linguagem simples e inclusiva para garantir a acessibilidade e a compreensão das informações fornecidas."
        )
    ) -> None:
        self.llm = llm
        self.tokenizador = tokenizador
        self.parametros_amostragem = parametros_amostragem
        self.system_prompt = system_prompt

    def _gerar(
        self,
        history: List[Dict[str, str]],
        prompt_usuario: str,
        texto_principal: str,
        contexto: str
    ) -> Iterator[List[Dict[str, str]]]:
        """
        Função de geração principal, chamada pelo Gradio.
        
        REFATORAÇÃO: Esta função foi otimizada. As três entradas (prompt_usuario,
        texto_principal, contexto) são agora combinadas em uma *única* mensagem
        de "role: user". Isso corrige um bug onde múltiplas mensagens "role: system"
        eram enviadas, quebrando o formato do chat template.
        """
        
        # --- 1. Logging da Interação ---
        interacao_usuario = {
            'prompt_sistema': self.system_prompt,
            'prompt_usuario': prompt_usuario,
            'texto_principal': texto_principal,
            'contexto': contexto
        }
        logging.info(f'interacao_usuario|{interacao_usuario}')

        # --- 2. Construção do Prompt do Usuário ---
        partes_prompt_usuario = []
        if prompt_usuario:
            partes_prompt_usuario.append(prompt_usuario)
            
        if contexto:
            instrucao_contexto = (
                'Instruções obrigatórias: Responda com base exclusivamente no contexto fornecido abaixo '
                'e evite informações adicionais. Segue o contexto:\n' + contexto
            )
            partes_prompt_usuario.append(instrucao_contexto)
        
        partes_prompt_usuario.append(texto_principal)
        
        # Combina todas as partes em uma única string
        prompt_completo_usuario = "\n\n".join(partes_prompt_usuario).strip()

        # --- 3. Preparação das Mensagens para o Modelo ---
        mensagens_para_modelo = []
        if not history:
            # Se for a primeira mensagem, adiciona o system prompt
            mensagens_para_modelo.append({"role": "system", "content": self.system_prompt})
        else:
            # Caso contrário, usa o histórico existente
            mensagens_para_modelo = history
        
        # Adiciona a mensagem completa do usuário atual
        mensagens_para_modelo.append({"role": "user", "content": prompt_completo_usuario})

        # --- 4. Atualização da UI (Gradio) ---
        # Adiciona a mensagem do usuário ao histórico de exibição e a exibe imediatamente
        history.append({"role": "user", "content": prompt_completo_usuario})
        yield history

        # Cria o placeholder da resposta do assistente
        response = {"role": "assistant", "content": ""}
        
        # --- 5. Geração do Modelo (Streaming) ---
        prompt_para_modelo = self.tokenizador.apply_chat_template(
            mensagens_para_modelo,
            tokenize=False,
            add_generation_prompt=True
        )

        texto_completo_gerado = ""
        prefixo_assistente = "<|start_header_id|>assistant<|end_header_id|>"

        for chunk in self.llm.gerar(prompt_para_modelo, self.parametros_amostragem):
            texto_completo_gerado = chunk.outputs[0].text
            
            # Limpa o prefixo do assistente que o modelo pode gerar
            if texto_completo_gerado.startswith(prefixo_assistente):
                response["content"] = texto_completo_gerado.removeprefix(prefixo_assistente).lstrip()
            else:
                response["content"] = texto_completo_gerado
            
            # Emite (yield) o histórico de exibição + a resposta parcial do assistente
            yield history + [response]


    # Funções para aplicar 'Gostei', 'Não Gostei', 'Desfazer' e 'Refazer' na interface
    def aplicar_gostei(self, data: gr.LikeData):
        """Captura feedback de 'Gostei' ou 'Não Gostei'."""
        if data.liked:
            print("like ", data.value)
        else:
            print("dislike ", data.value)

    def aplicar_desfazer(self, history, undo_data: gr.UndoData):
        """Desfaz a última interação no histórico do chatbot."""
        return history[:undo_data.index], history[undo_data.index]['content']

    def aplicar_refazer(self, history, prompt_usuario, texto_principal, contexto):
        """Refaz uma interação específica no histórico do chatbot."""
        historico_anterior = history[:-1]  # Refazer a partir da penúltima interação
        ultimo_prompt = history[-1]['content']
        
        # NOTA: Esta lógica de refazer pode não funcionar como esperado
        # com a nova estrutura de prompt. Pode precisar de ajuste.
        yield from self._gerar(historico_anterior, ultimo_prompt, texto_principal, contexto)

    def processar_feedback(self, mensagens, like_data: gr.LikeData, estado):
        """Processa o feedback do usuário (like/dislike) e loga."""
        dados_feedback = {
            "mensagens": mensagens,
            "resposta_avaliada": like_data.value,
            "indice_mensagem": like_data.index,
            "curtiu": like_data.liked,
            **estado  # Incorpora as chaves e valores do estado
        }
        logging.info(f"avaliacao|{dados_feedback}")

    def atualizar_estado(self, request: gr.Request, estado):
        """Atualiza o estado local com informações do usuário e ip."""
        ip_cliente = request.client.host
        usuario = request.username or "Desconhecido"

        estado["client_ip"] = ip_cliente
        estado["usuario_logado"] = usuario

        dict_estado = {'usuario': usuario, 'ip': ip_cliente}
        logging.info(f'logging_usuario|{dict_estado}')
        return None, None

    def iniciar(self):
        """Configura e lança a interface Gradio."""

        # ------ TEMA E CSS DA INTERFACE (idênticos ao original) ------
        tema = gr.themes.Default(
            primary_hue="blue", secondary_hue="green", neutral_hue="slate",
            font=["Arial", "sans-serif"], text_size="md",
        )
        
        # (CSS Omitido por concisão - é idêntico ao original)
        estilo = """ ... """ 

        with gr.Blocks(theme=tema, css=estilo) as interface_chat:
            
            with gr.Row():
                gr.Image(
                    value='/ouvdia/assets/ouvdia_logo.jpeg', type="filepath",
                    label='OuvdIA', show_label=False, interactive=False,
                    elem_id="image-1"
                )
                gr.Image(
                    value="/ouvdia/assets/grafos_capa_3.jpeg", type="filepath",
                    label="Assistente Inteligente", show_label=False, interactive=False,
                    elem_id="image-2"
                )

            interface_chat.css = """
            #image-1 { flex: 1; max-width: 14%; height: auto; }
            #image-2 { flex: 4; max-width: 86%; height: auto; }
            """

            gr.Markdown("# 💡OuvdIA - Assistente Inteligente da Ouvidoria da RFB")

            with gr.Row():
                prompt_usuario = gr.Textbox(
                    label='1). Prompt (preenchimento obrigatório): Instruções para a OuvdIA',
                    placeholder='📝 Insira aqui as instruções...'
                )
                texto_principal = gr.Textbox(
                    label='2). Texto Principal (preenchimento obrigatório)',
                    lines=1,
                    placeholder='📝 Insira aqui o texto...'
                )
                contexto = gr.Textbox(
                    label='3). Contexto (opcional)',
                    placeholder='📝 Insira aqui o contexto...'
                )

            chatbot = gr.Chatbot(
                label="Histórico do Chat",
                type="messages",
                avatar_images=(
                    '/ouvdia/assets/avatar_usuario.png',
                    '/ouvdia/assets/avatar_chatbot.svg'
                ),
            )

            estado = gr.State({"client_ip": None, "usuario_logado": None})
            ip_info = gr.Markdown()
            user_info = gr.Markdown()

            interface_chat.load(self.atualizar_estado, inputs=[estado], outputs=[ip_info, user_info])

            chatbot.like(
                fn=self.processar_feedback,
                inputs=[chatbot, estado],
                outputs=[]
            )
            
            chatbot.undo(self.aplicar_desfazer, chatbot, [chatbot, texto_principal])
            
            # (Retry opcional)
            # chatbot.retry(self.aplicar_refazer, [chatbot, prompt_usuario, texto_principal, contexto], [chatbot])

            texto_principal.submit(
                self._gerar,
                [chatbot, prompt_usuario, texto_principal, contexto],
                [chatbot],
                queue=True
            )
            
            # Limpa o campo texto_principal depois de enviar
            texto_principal.submit(lambda: "", None, [texto_principal])

        return interface_chat


# ========== Exemplo de Uso (Ponto de Entrada) ==========

if __name__ == "__main__":

    # ========== Inicialização do modelo e da interface ==========

    # Inicializa o modelo LLM
    llm = ModeloLinguagemContinuo(
        modelo="meta-llama/Llama-3.1-8B-Instruct",
        quantizacao=None,
        tipo_dados="float16",
        # Define limite explícito para o comprimento da sequência
        max_model_len=32768,
        # Aumentar a utilização de memória da GPU
        gpu_memory_utilization=0.95,
        max_num_seqs=256
    )

    # Obtém o tokenizador do modelo
    tokenizador = llm.llm_motor.tokenizer.tokenizer

    # Define os parâmetros de amostragem
    parametros_amostragem = SamplingParams(
        temperature=0.2,
        top_p=0.7,
        max_tokens=8192,
        stop_token_ids=[
            tokenizador.eos_token_id,
            tokenizador.convert_tokens_to_ids("<|eot_id|>")
        ],
    )

    # Inicializa a classe da interface
    interface_usuario = InterfaceUsuario(llm, tokenizador, parametros_amostragem)

    # Cria o Gradio Blocks (interface principal)
    interface_pronta = interface_usuario.iniciar()

    # ========== Lançar a interface sem autenticação ==========
    imagem_icone_url = '/ouvdia/assets/rfb_logo_1.ico'
    
    print("Iniciando a interface Gradio em http://0.0.0.0:7860")
    
    interface_pronta.launch(
        share=False,
        # 'auth=usuarios_validos'
        auth_message="💡 Bem-vindo(a) ao Chatbot da OuvdIA – Suíte de Agentes Inteligentes da Ouvidoria da RFB!",
        #favicon_path=imagem_icone_url,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )