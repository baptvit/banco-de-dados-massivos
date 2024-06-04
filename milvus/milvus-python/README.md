## README.md para Aplicação Python Milvus com Docker e PySpark

### Visão Geral

Este repositório fornece uma aplicação Python para interagir com o banco de dados de vetores Milvus usando o Docker. A aplicação inicial demonstra operações básicas de conexão, criação de coleções, inserção de dados, criação de índices, busca semântica, consultas e exclusão de entidades.

### Pré-requisitos

1. **Docker e Docker Compose:** Instale e configure o Docker e o Docker Compose para gerenciar contêineres.

2. **Milvus Standalone:** Tenha o Milvus Standalone em execução na porta 19530 no localhost.

### Instruções

1. **Gerenciamento de Dependências:**

   Utilize o Poetry para gerenciar as dependências do projeto.

   ```bash
   poetry install
   ```

   Ative o ambiente virtual criado pelo Poetry para executar os scripts:

   ```bash
   poetry env info
   ```

2. **Executando a Aplicação:**

   Execute o script principal usando o Poetry:

   ```bash
   poetry run python ./milvus_python/hello_milvus.py
   ```

### Próximos Passos

O projeto será expandido para incluir:

1. **Carga automatizada de dados:** Usando PySpark para carregar dados em massa no Milvus.

2. **Criação de diferentes índices:** Experimentação com diferentes tipos de índices para otimizar a busca.

3. **Scripts de consulta para benchmarks:** Avaliação do desempenho da busca com diferentes consultas.

### Contribuições

Sinta-se à vontade para contribuir com sugestões, melhorias ou novas funcionalidades.

### Licença

Este projeto está sob a Licença MIT.

### Contato

Para dúvidas ou sugestões, entre em contato conosco.