# Roteiro de demonstração — Predição de Churn

## Apresentação final (12/11/2026)

**Projeto:** Sistema Inteligente para Predição de Churn em Telecomunicações  
**Equipe:** Antonio Vicente, Guilherme de Abreu, João Emanuel Dirceu e Agenor Neto  
**Base:** Iranian Churn Dataset (UCI), com 3.150 linhas originais; 2.850 registros únicos após exclusão de 300 duplicatas exatas.

### Sequência sugerida (8–10 minutos)

1. **Vicente — problema e solução (1 min).** Churn é o abandono do serviço. Demonstrar como dados comportamentais podem apoiar a identificação antecipada de risco e ações de retenção. Mostrar tela de visão geral.
2. **Abreu — dados e limpeza (2 min).** Explicar os 13 atributos, a coluna-alvo, 300 duplicatas exatas e o cuidado contra vazamento: registros com preditores idênticos permanecem no mesmo grupo para particionamento. Exibir Dashboard e distribuições reais.
3. **Dirceu — IA e avaliação (2 min).** Comparar Regressão Logística (baseline) e Random Forest no mesmo conjunto de teste; descrever F1-score, ROC-AUC, matriz de confusão e importância das variáveis. Exibir página Modelos.
4. **Agenor — demonstração e conclusão (3 min).** Na tela Prever Churn, carregar um exemplo real, selecionar Random Forest e executar a predição. Alterar o limiar de decisão e executar novamente; mostrar que a probabilidade não muda, mas a classificação pode mudar. Testar também Regressão Logística e concluir com limites e próximos passos.
5. **Perguntas (2 min).** Manter os resultados de teste e as limitações em mãos.

### Resultados do teste reproduzível

Execução do GitHub Actions, em 10/11/2026, com CSV oficial da UCI e branch de integração:

- Partição: 2.279 registros de treinamento e 571 de teste, com separação por grupos.
- **Random Forest:** F1 = 0,9081; ROC-AUC = 0,9808; Average Precision = 0,9321.
- Matriz Random Forest no teste (linhas = real, colunas = previsto; ordem 0,1): [[470,11],[6,84]].
- **Regressão Logística:** F1 = 0,6160; ROC-AUC = 0,9356; Average Precision = 0,7726.
- Matriz Regressão Logística: [[389,92],[9,81]].
- Evidência: https://github.com/vicenteosorioneto/iranian-churn-backend/actions/runs/42927537239

**Importante:** os resultados não devem ser descritos como desempenho em produção nem como garantias de acerto para outros países ou empresas. A nova separação por grupos pode produzir resultados diferentes dos documentados na P1.

### Demonstração técnica

Backend:
~~~powershell
python -m app.ml.train
python -m scripts.demo_smoke
uvicorn app.main:app --reload
~~~

Frontend:
~~~powershell
npm run dev
~~~

Abrir http://localhost:3000. Percorrer Visão Geral → Modelos → Prever Churn → Análise dos Dados.

### Perguntas da banca

**Qual a contribuição?** Uma aplicação acadêmica de análise e previsão de churn que combina preparo reproduzível, comparação de dois classificadores, avaliação com separação por grupos e interface de apoio à interpretação.

**Por que Random Forest?** Permite modelar relações não lineares; superou a regressão logística no conjunto de teste desta execução, mas deve ser validado em outros cenários.

**Como evitar vazamento de dados?** Remoção de linhas exatamente duplicadas; agrupamento por atributos preditores antes de separar treino/teste; preprocessadores ajustados somente no treino.

**A acurácia sozinha basta?** Não. O alvo é desbalanceado, por isso F1, ROC-AUC, Average Precision, recall, precisão e matriz de confusão são apresentados.

**O que o limiar significa?** O valor de corte aplicado à probabilidade produzida pelo modelo. Baixá-lo tende a identificar mais clientes potencialmente em risco, com possíveis falsos positivos adicionais.

**Importância de atributos prova causa?** Não. Importância por permutação indica associação com desempenho preditivo no conjunto de teste, não relação causal.

**Quais os limites?** Uma única base pública de telecomunicações iraniana; sem validação externa; hipóteses de pré-processamento; grupos conflitantes; sem intervenção real de retenção; probabilidades não calibradas especificamente para decisões comerciais.

### Verificação antes de apresentar

- [ ] Frontend e backend funcionando sem erros.
- [ ] CSV oficial presente em data/raw/iranian_churn.csv (não versionar o arquivo).
- [ ] Modelos treinados e métricas geradas após a atualização.
- [ ] Página Modelos exibe números, curvas e matrizes.
- [ ] Tela Prever Churn apresenta pelo menos uma predição real.
- [ ] A opção de trocar o modelo funciona.
- [ ] Os exemplos reais e o limiar funcionam.
- [ ] Mostrar um slide com limitações e conclusões.
- [ ] Gravar vídeo de reserva da demonstração.
