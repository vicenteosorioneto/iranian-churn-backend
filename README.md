# Iranian Churn Backend

Backend acadêmico para preparar a Iranian Churn Dataset, treinar e comparar modelos de classificação e, após o treinamento, oferecer previsões por uma API FastAPI.

## Arquitetura

- `app/api`: rotas HTTP e contratos Pydantic.
- `app/ml`: schema do dataset, pré-processamento, treinamento, avaliação e inferência.
- `app/services`: integração entre os artefatos treinados e a API.
- `data/raw`: CSV original (não versionado).
- `data/processed`: espaço reservado para dados derivados.
- `models`: pipelines completos serializados com Joblib.
- `reports`: métricas e futuras figuras.
- `tests`: testes unitários e da API.

O treinamento remove 300 duplicatas exatas (3.150 → 2.850 registros) e utiliza `StratifiedGroupKFold` com 5 partições (`random_state=42`), reservando uma para teste. Os grupos são definidos por atributos idênticos para evitar que combinações iguais de features apareçam simultaneamente em treino e teste. Todo transformer é ajustado apenas por meio do `fit` do pipeline sobre o conjunto de treino, evitando vazamento de dados.

## Atributos

As 13 features estão centralizadas em `app/ml/features.py`:

| Grupo | Atributos |
| --- | --- |
| Numéricos | Call Failure, Subscription Length, Seconds of Use, Frequency of use, Frequency of SMS, Distinct Called Numbers, Age, Customer Value |
| Binários/categóricos | Complains, Tariff Plan, Status |
| Ordinais | Charge Amount, Age Group |
| Target | Churn |

Embora a documentação da fonte descreva `Charge Amount` como ordinal de 0 a 9,
o CSV oficial analisado contém 7 registros no nível 10. O dado bruto é preservado
e o contrato usa o domínio ordinal observado de 0 a 10; os valores 10 não são
substituídos nem excluídos.

### Decisões metodológicas pendentes

Na base original, `Churn = 0` representa 2.655 registros (84,29%) e `Churn = 1`
representa 495 (15,71%), caracterizando desbalanceamento da variável alvo. Não é
aplicado SMOTE ou qualquer oversampling. Os estimadores aceitam a estratégia
`class_weight="balanced"`, preservando o pipeline como ponto único de configuração.

Também existem 300 linhas excedentes exatamente duplicadas e grupos de features
idênticas com targets conflitantes. A inspeção relata esses casos; o treinamento remove
somente duplicatas exatas, mas não apaga grupos com rótulos conflitantes. A separação
por grupos mantém combinações idênticas de features fora de conjuntos distintos.

Na Regressão Logística, valores numéricos ausentes recebem a mediana e os valores são padronizados com `StandardScaler`. No Random Forest, a imputação é igual, mas não há escala. Em ambos, atributos binários/categóricos usam imputação pela moda e `OneHotEncoder`; os ordinais usam imputação pela moda e mantêm sua representação ordenada. Cada estimador e seu pré-processamento formam um único `Pipeline`.

## Instalação

Requer Python 3.11 ou superior.

```bash
python -m venv .venv
```

No PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset e treinamento

Coloque o CSV real neste caminho exato:

```text
data/raw/iranian_churn.csv
```

O arquivo deve conter as 13 features listadas acima e a coluna `Churn`. O carregador remove espaços nas extremidades dos nomes das colunas, valida o schema e informa claramente qualquer coluna ausente. Nenhum dado sintético é usado.

Execute o treinamento a partir da raiz:

```bash
python -m app.ml.train
```

Antes do treinamento, inspecione dimensões, tipos, valores ausentes, duplicados,
domínios, balanceamento e estatísticas sem alterar os dados:

```bash
python -m app.ml.inspect_data
```

Esse comando gera:

- `models/logistic_regression.joblib`;
- `models/random_forest.joblib`;
- `reports/metrics.json` com Accuracy, Precision, Recall, F1, ROC-AUC, matriz de confusão e pontos da curva ROC;
- `reports/feature_importance.json` com importância por permutação no conjunto de teste para o Random Forest.

## API

Inicie o servidor:

```bash
uvicorn app.main:app --reload
```

Rotas iniciais:

- `GET /api/health` — disponibilidade da aplicação;
- `GET /api/models/metrics` — métricas geradas pelo treinamento;
- `POST /api/predict` — previsão com um pipeline treinado;
- `GET /api/analytics/summary` — indicadores e distribuição observada de churn para dashboard;
- `GET /api/models/feature-importance` — importância de variáveis medida no conjunto de teste.

A documentação interativa fica em `http://127.0.0.1:8000/docs`. As rotas de métricas e previsão retornam erros explícitos enquanto os artefatos reais ainda não existirem.

## Testes

```bash
pytest
```

## Estrutura

```text
app/
  api/
    routes/          # health, métricas e previsão
    schemas/         # contratos da API
  ml/                # dados, features, pipelines, treino e avaliação
  services/          # serviço de carregamento e inferência
data/
  raw/
  processed/
models/
notebooks/
reports/
  figures/
tests/
Dockerfile
requirements.txt
```

## Demonstração integrada com Next.js

1. Obtenha a base original Iranian Churn na UCI, mantenha seu conteúdo e salve como `data/raw/iranian_churn.csv`.
2. Instale dependências e execute `python -m app.ml.inspect_data`.
3. Treine os dois modelos: `python -m app.ml.train`. Este comando produz os arquivos locais ignorados pelo Git, as métricas, curvas ROC e as importâncias por permutação.
4. Execute `pytest` e depois `uvicorn app.main:app --reload`.
5. Em outra janela, execute o frontend Next.js na porta 3000, com `NEXT_PUBLIC_API_URL=http://localhost:8000/api`.

O navegador local está autorizado via CORS para `http://localhost:3000` e `http://127.0.0.1:3000`. Para outras origens, defina `CHURN_CORS_ORIGINS` como uma lista separada por vírgulas. Não use `*` em produção.

Os dados exibidos são agregados dos registros únicos da base; nenhum gráfico, pontuação, matriz de confusão ou probabilidade é inventado. A predição precisa dos artefatos Joblib produzidos pelo treinamento. O serviço devolve erros explícitos quando os dados ou modelos estão ausentes.
