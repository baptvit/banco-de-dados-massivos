## ChormaDB Standalone Dockerizado: Consultas Semânticas em Dados Vectorizados

### Introdução

Este repositório fornece uma implementação simplificada do ChormaDB Standalone utilizando o Docker e o Minion para garantir persistência de dados. O ChormaDB é um banco de dados vetorial de código aberto que permite realizar consultas semânticas em dados vectorizados, abrindo um leque de possibilidades para aplicações de aprendizado de máquina e processamento de linguagem natural.

### Projeto

O projeto assume que o ChormaDB estará executando na porta 8001 e utiliza persistência ativa para armazenar dados e índices criados nas collections. A imagem Docker utilizada é a `chromadb/chroma:0.5.0`, fornecida pelos desenvolvedores do chromadb.

Na pasta `chroma-data` estaram todos os dados, indexs e recursos utilizados para persistir os dados no banco.

### Pré-requisitos

1.  **Docker:** Instale o Docker seguindo as instruções em [https://docs.docker.com/](https://docs.docker.com/).
2.  **Docker Compose:** Instale o Docker Compose seguindo as instruções em [https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/).

### Construindo e Executando

Para iniciar o container Docker do ChormaDB Standalone, execute o seguinte comando:

```bash
docker-compose up
```

Para pausar o container, execute:

```bash
docker-compose down
```

### Próximos Passos

Este repositório fornece uma base para iniciar com o Milvus Standalone. Para mais informações sobre o Milvus e como realizar operações específicas, consulte a documentação oficial: [https://milvus.io/docs](https://milvus.io/docs).

### Contribuições

Se você tiver sugestões de melhorias ou quiser contribuir para o projeto, sinta-se à vontade para abrir um issue ou pull request.
