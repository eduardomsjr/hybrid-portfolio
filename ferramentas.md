Todas as ferramentas utilizadas para o desenvolvimento da pesquisa são de código aberto. 

A linguagem de programação utilizada foi **Python**, bastante empregada em projetos de ciências de dados por conter módulos e bibliotecas específicos para esta tarefa.

Para a modelagem com VAR, correlação, teste de *Dickey-Fuller Aumentado*, causalidade de *Granger* e teste de cointegração de *Johansen* foi utilizado o módulo **statsmodels**, que fornece classes e funções para a estimativa de diversos modelos estatísticos, bem como para a realização de testes e exploração estatística de dados (Seabold e Perktold, 2010). 

Para realização da otimização de média variância e *Black-Litterman*, a biblioteca de modelagem para problemas de otimização convexa **CVXPY** foi utilizada *(Diamond e Boyd, 2016)*. Para análise do desempenho e risco dos portfólios gerados, foi utiliza a biblioteca **pyfolio-reloaded**. A criação dos gráficos foi realizada com as bibliotecas **Plotly**, **Seaborn** e **Matplotlib**. O módulo **NumPy**, para a manipulação de matrizes, e **Pandas**, para a criação de dataframes, também foram utilizados. 

Por fim, para a construção do aplicativo WEB, o framework **Streamlit** foi usado.

- https://www.python.org
- https://pyfolio.ml4trading.io
- https://plotly.com/python
- https://seaborn.pydata.org
- https://matplotlib.org
- https://numpy.org
- https://pandas.pydata.org
- https://streamlit.io