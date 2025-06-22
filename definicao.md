Para que seja possível avaliar os resultados, cinco portfólios de exemplo serão criados, bem como será demonstrado o desenvolvimento da estratégia. Os cinco portfólios criados têm cada um as seguintes características:

- Portfólio 1: Muita ênfase aos critérios ESG com alto risco;
- Portfólio 2: Muita ênfase aos critérios ESG com baixo risco;
- Portfólio 3: Pouca ênfase aos critérios ESG com alto risco;
- Portfólio 4: Pouca ênfase aos critérios ESG com baixo risco;
- Portfólio 5: Ênfase média aos critérios ESG com risco médio.

Ainda que seja possível excluir alguma ação ou setor, não foi utilizada essa possibilidade no comparativo.

A estratégia para comparar os portfólios tem o seguinte desenvolvimento:

- Na primeira fase, o histórico de preços ajustados de todas as ações que compõem o índice do ETF ESGB11, além do ETF BOVA11 representando o benchmark, é obtido no período de 2/1/2019 a 1/8/2024;
- Na segunda fase, as ações obtidas são agrupadas por setor;
- Na terceira fase, os retornos logarítmicos dos preços ajustados das ações selecionadas do período que compreende de 2/1/2019 a 4/1/2021 são colocados como entrada do modelo VAR para que possa aprender e realizar as previsões;
- Na quarta fase, o modelo VAR realiza a previsão dos retornos do período de 5/1/2021 a 5/1/2023 das ações agrupadas por setor (fase 2);
- Na quinta fase, que inicia as fases de interação com o investidor, são escolhidos os critérios ESG e o risco para a criação do portfólio; 
- Na sexta fase, com base nos critérios ESG e risco obtidos na fase 5, são definidas as pontuações mínimas para a escolha das ações;
- Na sétima fase, os retornos previstos do modelo VAR (fase 4) são entregues como entrada do modelo Black-Litterman que faz a alocação dos ativos, criando cada portfólio baseado nos critérios ESG e risco escolhidos pelo investidor (fase 5), estes definidos pelo modelo de otimização de média variância (utilizando as pontuações obtidas durante a sexta fase);
- Para fins de comparação, na oitava e derradeira fase, também é criado um portfólio com retornos VAR com as mesmas ações escolhidas para o modelo Black-Litterman baseadas nos critérios ESG e risco escolhidos. Todavia, diferentemente do modelo BL, o portfólio tem pesos iguais. 
- Os portfólios criados são comparados com o benchmark no período de 6/1/2023 a 1/8/2024 e os resultados são apresentados.