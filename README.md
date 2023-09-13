# CHATBOT PDF

<a href="arq.png"></a>

## Construção da base de conhecimento

1. Extrair o conteúdo do PDF
2. Dividir o conteúdo em Chunks
3. Realizar o embedding das chunks (Transformar em um vetor)
4. Salvar os vetores em uma base de conhecimento

## Interação do usuário

1. Realiza a pergunta (query)
2. Realiza o embedding da query
3. Extração do sentido semântico da query
4. Busca pelos chunks que coincidem com a semântica da query
