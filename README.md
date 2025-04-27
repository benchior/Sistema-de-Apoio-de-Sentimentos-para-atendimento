# Sistema de Análise de Sentimentos para Atendimento ao Cliente

## Visão Geral
Este projeto é um sistema completo de análise de sentimentos projetado para feedback de atendimento ao cliente. Inclui processamento avançado de linguagem natural, modelos de machine learning, dashboard interativo, priorização automática e integração com APIs de CRM.

## Funcionalidades
- Análise de sentimentos usando combinação de métodos de machine learning e regras
- Extração de palavras-chave e determinação do nível de prioridade
- Dashboard interativo construído com Dash para análise em tempo real e visualização de dados
- Integração com sistemas CRM para envio de alertas de prioridade
- Suporte a logging e configuração via variáveis de ambiente

## Instalação

1. Clone o repositório:
   ```
   git clone <url_do_repositorio>
   cd sentiment_analysis_system
   ```

2. Crie e ative um ambiente virtual no Windows 11:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Baixe o modelo de inglês do spaCy:
   ```
   python -m spacy download en_core_web_md
   ```

5. Configure as variáveis de ambiente no Windows:
   - Copie `.env.example` para `.env`
   - Edite o arquivo `.env` para adicionar sua URL e chave da API do CRM
   - Para definir variáveis de ambiente temporariamente no terminal, use:
     ```
     set CRM_API_URL=https://sua-api-crm.com/api
     set CRM_API_KEY=sua_chave_api_aqui
     ```
   - Para definir variáveis permanentemente, use as configurações do sistema do Windows

## Uso
   python -m spacy download en_core_web_md
   pip install -r requirements.txt

Execute a aplicação principal:
```
python -m sentiment_analysis.main
```

Isso iniciará o dashboard e realizará análises de sentimento de exemplo.

## Estrutura do Projeto

```
sentiment_analysis_system/
├── sentiment_analysis/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── dashboard.py
│   ├── crm.py
│   └── main.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
