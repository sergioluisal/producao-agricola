# Dashboard de Acompanhamento

Este dashboard foi desenvolvido para visualizar dados de acompanhamento de pedidos/suprimentos de forma interativa, com tratamento robusto de erros e divisão por zero.

## 🚀 Funcionalidades

### 📊 Visualizações Disponíveis
- **Métricas Principais**: Total de pedidos, entregues, quantidade total e taxa de entrega
- **Gráficos de Barras**: Top 10 estados, tipos de produto, modelos
- **Gráficos de Pizza**: Distribuição por status e entrega
- **Linha Temporal**: Evolução dos pedidos ao longo do tempo
- **Mapa Interativo**: Distribuição geográfica por estado (se dados disponíveis)

### 🔍 Filtros Interativos
- **Estado**: Filtrar por UF
- **Status**: Filtrar por status atual
- **Tipo de Produto**: Filtrar por categoria
- **Status de Entrega**: Filtrar por entregues/pendentes

### 🛡️ Tratamento de Erros
- **Divisão por Zero**: Todas as operações matemáticas são protegidas
- **Dados Ausentes**: Tratamento de valores nulos e colunas inexistentes
- **Encoding**: Suporte a múltiplas codificações de arquivo
- **Dados Vazios**: Mensagens informativas quando não há dados

## 📋 Campos Suportados

O dashboard detecta automaticamente as colunas disponíveis e adapta as visualizações:

- `NumeroPedido`
- `DataPedido`
- `ModeloProduto`
- `TipoProduto`
- `QuantidadeProduto`
- `OrdemServico`
- `NumeroSerie`
- `ApelidoDoEquipamento`
- `StatusAtual`
- `PrevisaoEntrega`
- `Entregue`
- `Uf` (para visualizações geográficas)
- `Municipio`

## 🚀 Como Usar

### Pré-requisitos
- Python 3.7 ou superior
- Arquivo CSV com dados de acompanhamento

### Instalação

1. **Extraia os arquivos** do projeto
2. **Navegue até a pasta** do projeto
3. **Crie um ambiente virtual:**
   ```bash
   python -m venv venv_acompanhamento
   ```
4. **Ative o ambiente virtual:**
   - Windows: `venv_acompanhamento\Scripts\activate`
   - Linux/Mac: `source venv_acompanhamento/bin/activate`
5. **Instale as dependências:**
   ```bash
   pip install -r requirements_acompanhamento.txt
   ```

### Execução

1. **Coloque seu arquivo CSV** na mesma pasta do script com o nome `acompanhamento.csv`
2. **Execute o dashboard:**
   ```bash
   streamlit run dashboard_acompanhamento_streamlit.py
   ```
3. **Acesse no navegador:** `http://localhost:8501`

## 📁 Estrutura de Arquivos

```
projeto/
├── dashboard_acompanhamento_streamlit.py  # Código principal
├── requirements_acompanhamento.txt        # Dependências
├── README_acompanhamento.md              # Esta documentação
└── acompanhamento.csv                    # Seus dados (você deve fornecer)
```

## 🔧 Personalização

### Adicionando Novas Visualizações
O código é modular e permite fácil adição de novos gráficos. Cada função de visualização inclui tratamento de erros.

### Modificando Filtros
Os filtros são gerados dinamicamente baseados nas colunas disponíveis. Para adicionar novos filtros, modifique a seção "Sidebar para filtros".

### Alterando Cores e Estilos
As cores dos gráficos podem ser personalizadas modificando os parâmetros `color_discrete_sequence` nas funções de criação de gráficos.

## 🐛 Solução de Problemas

### Erro de Encoding
O dashboard tenta múltiplas codificações automaticamente. Se ainda houver problemas, salve seu CSV em UTF-8.

### Dados Não Aparecem
Verifique se:
- O arquivo se chama `acompanhamento.csv`
- Está na mesma pasta do script
- As colunas têm os nomes esperados

### Performance
Para grandes volumes de dados, considere:
- Filtrar dados antes de carregar
- Usar amostragem para visualizações
- Otimizar consultas pandas

## 📞 Suporte

Se encontrar problemas:
1. Verifique se todas as dependências estão instaladas
2. Confirme que o arquivo CSV está no formato correto
3. Verifique os logs no terminal para mensagens de erro específicas

## 🔄 Atualizações

Para atualizar o dashboard:
1. Substitua o arquivo Python principal
2. Atualize as dependências se necessário
3. Reinicie o Streamlit

