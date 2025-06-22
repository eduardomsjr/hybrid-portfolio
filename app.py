#import warnings
#warnings.filterwarnings("ignore")

import contextlib
import os

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pyfolio as pf

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import cvxpy as cp
from cvxpy.atoms.affine.wraps import psd_wrap
from statsmodels.stats.correlation_tools import cov_nearest  
import plotly.graph_objects as go

st.set_page_config(
    page_title='Portfólio ESG - Black-Litterman com previsões VAR',
    page_icon='💲',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'Get Help': 'https://github/eduardomsjr',
        'Report a bug': 'https://github/eduardomsjr/hybrid-portfolio',
        'About': "Feito por Eduardo Jr."
    }
)

# Função para calcular médias móveis
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

# Função para teste de causalidade de Granger
maxlag=4
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test'):
    """Verificar a Causalidade de Granger de todas as combinações possíveis da série temporal.
       As linhas são a variável de resposta, as colunas são os preditores. Os valores na tabela são os valores-p. 
       Valores-p menores que o nível de significância (0,05) implicam que a Hipótese Nula de que os coeficientes 
       dos valores passados ​​correspondentes são zero, ou seja, X não causa Y, pode ser rejeitada.

       dados: dataframe do Pandas contendo as variáveis ​​da série temporal
       variáveis: lista contendo os nomes das variáveis ​​da série temporal.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

# Função para teste de cointegração (Johansen)
def cointegration_test(df, alpha=0.05): 
    """Realizar o Teste de Cointegração de Johansen e mostrar sumário"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    # Summary
    st.write('Ação       ::       Teste Estatístico > C(95%)       =>       Significado  \n')
    for col, trace, cvt in zip(df.columns, traces, cvts):
        st.write((col), ':: ', (round(trace,2), 9), ">", (cvt, 8), ' =>  ' , trace > cvt)

# Função para teste ADF 
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Realizar ADF para testar estacionariedade de uma série e mostrar sumário"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']         
    st.write('Teste Dickey-Fuller Aumentado (ADF) em', name)    
    if p_value <= signif:
        st.write('=> P-Valor = ', p_value, '. Rejeitando Hipótese Nula.')
        st.write('=> Série é Estacionária.')
    else:
        st.write('=> P-Valor = ', p_value, '. Fraca evidência para rejeitar a hipótese nula.')
        st.write('=> Série não é Estacionária.')

# Função para realizar previsão dos retornos usando VAR 
def var_forecast(df, obs=496):
    # Cria dataframe vazio para guardar os dados forecast
    df_forecast = pd.DataFrame(index=df[obs-1:].head(1).index, columns=df.columns + '_previsto')
    for n_obs in range(obs,996):
        # Divide datraframe em treinamento e teste
        df_forec_train = df[0:n_obs]        
        # Criação do modelo VAR
        model = VAR(df_forec_train)
        model_fitted = model.fit(1)
        forecast_input_ = df_forec_train.values[-1:]
        #Previsão do modelo
        fc = model_fitted.forecast(y=forecast_input_, steps=1)
        # Transforma numpy array em dataframe
        df_fc = pd.DataFrame(fc, index=df[n_obs:].head(1).index, columns=df.columns + '_previsto')
        # Concatena os dataframes
        df_forecast = pd.concat([df_forecast, df_fc])
    # Limpa os NAs
    df_forecast.dropna(inplace=True)       
    return df_forecast

# Função para Black-Litterman (retornos e matriz de covariância):
def black_litterman(prices, data, A):
    '''
        Parâmetros
        ----------
        prices: Pandas DataFrame
        Tabela de símbolos de ações com preços diários de ações por um período.
        data: Série Pandas
        Tabela de coluna única com dados de capitalização de mercado para cada ação.
        A: float
        Parâmetro de aversão ao risco variando de 0 a 5.

        Retornos
        -------
        Retornos posteriores de Black-Litterman e matriz de covariância de ativos fornecidos usando a
        Metodologia de Atílio Meucci. 2 Pandas DataFrames.

    '''    
    #Calcular retornos logarítmicos médios anuais:
    stock_log_ret = prices
    avg_yearly_ret = stock_log_ret.mean() * 252
    
    #Vetor de visões (Q) como retornos logarítmicos médios:
    Q = avg_yearly_ret[avg_yearly_ret>-1]
    data = data.loc[Q.index, :]
    
    stock_log_ret = stock_log_ret.loc[:, Q.index]
    
    #Pesos baseados no Market Cap para portfólio de mercado:
    mcap_wgts = (data.Market_Cap / data.Market_Cap.sum()).values #NP array for calculations to come
    
    #Matriz de covariância dos retornos logarítmicos (S):
    S = stock_log_ret.cov()
    
    #Assegurar que não há NaNs: 
    S.fillna(S.mean(), inplace=True)          #Preencher com a média de cada ação
    S.fillna(S.mean().mean(), inplace=True)   #Para ações com média 0, preencher com média geral
    
    #Vetor de retornos de excesso de equilíbrio implícito (pi = 2A*S*w -> Meucci): 
    pi = 2.0*A*(S @ mcap_wgts)
    
    #Matriz identidade (P) com 1s mostrando a posição da ação para essa visão (previsão de retornos):
    P = np.zeros((len(Q), len(Q)))   #Matrix com tamanho igual à quantidade de ações e visões
    np.fill_diagonal(P, 1)           #Diagonal da matriz com 1 

    #Escalar (tau) e matriz de incerteza das visões (ômega):
        #tau entre 0 e 1 --> 1/tamanho da série temporal - Meucci
        #c = 1 - Meucci  --> confiança geral constante no estimador de retorno das visões
        #ômega = 1/c * P * S * P^T - Meucci
    tau = 1.0/float(len(prices))
    c = 1.0
    omega = np.dot(np.dot(P, S), P.T) / c 
    
    #Excesso dos Retornos de BL - Meucci
        # = pi + tau*S*P^T * (tau*P*S*P^T + omega)^-1 * (Q - P*pi)
    r2 = np.linalg.inv(tau*P@S@P.T + omega)
    post_pi = pi + np.dot((tau*S@P.T) @ r2, (Q - P@pi))
    
    #Matriz de Covariância BL - Meucci
        # = (1+tau)*S - tau^2*S*P.T * (tau*P*S*P.T + omega)^-1 * P*S
    c2 = np.linalg.inv(tau*P@S@P.T + omega) 
    post_S = (1.0+tau)*S - np.dot(np.dot(tau**2.0*np.dot(S, P.T), c2), np.dot(P, S))
    
    #Assegurar matriz simétrica e positivamente estrita semi denifinida
    sym_S = (post_S + post_S.T) / 2
    semidef_S = cov_nearest(sym_S)
    
    return post_pi, semidef_S

# Função para otimização de Markowitz usando os retornos e covariância de Black-Litterman
def allocate_capital(E_scr, S_scr, G_scr, A, sectors=None, stocks=None):
    '''
    Parâmetros
    ----------
    E_scr: INT
    Pontuação (0-100) para importância ambiental, 100 significando maior importância.
    S_scr: INT
    Pontuação (0-100) para importância social, 100 significando maior importância.
    G_scr: INT
    Pontuação (0-100) para importâncial com governança, 100 significando maior importância.
    A: FLOAT
    Parâmetro de propensão ao risco do usuário (0-5), 5 propenso a maior risco.
    Setores: LISTA DE STRINGS
    Nomes dos setores.
    Ações: LISTA DE STRINGS
    Tickers das ações.

    Retornos
    -------
    Pandas DataFrame com as alocações, setores, betas e retornos para cada
    ação, juntamente com um dicionário contendo a variância do portfólio e as pontuações ESG
    (E, S, G, ESG total).

    Algoritmo de alocação de ativos baseado em Markowitz e na Teoria Moderna de Portfólios.
    Alocações de média variância com restrições de alocação e restrições de pontuação ESG 
    para exibir a preferência do usuário.

    '''
    #Preços das Ações:
    prices_path = './stock_prices.xlsx'
    prices = pd.read_excel(prices_path, engine='openpyxl')
    prices.set_index('Date', inplace=True)

    #Pontuações ESG:
    esg_path = "./stock_data.xlsx"
    data = pd.read_excel(esg_path, engine='openpyxl')
    data.set_index('Ticker', inplace=True)
    
    #Filtrar setores a evitar:
    if sectors != None:
        data = data.loc[data['Sectors'].isin(sectors) == False].sort_index()
        prices = prices.loc[:, data.index]
    
    #Filtrar ações a evitar:
    if stocks != None:
        data = data.loc[data.index.isin(stocks) == False].sort_index()
        prices = prices.loc[:, data.index]
    
    #Retornos e matriz de covariância BL
    ret, cov = black_litterman(prices, data, A)
    data = data.loc[ret.index]
    cov = psd_wrap(cov)                           #Matriz positiva semi definida
    
    #Definição de variáveis necessárias para otimização
    allocations = cp.Variable(len(ret))           #Variável a otimizar
    
    E = data.E_Score.values @ allocations         
    S = data.S_Score.values @ allocations         
    G = data.G_Score.values @ allocations         
    esg = data.ESG_Score.values @ allocations     
    
    var = cp.quad_form(allocations, cov)          #Variância/Risco a minimizar
    
    # Restrições: a soma das alocações precisa ser 1 (1), 
    # as ações só podem receber um máximo de 10% cada (2), 
    # nenhuma venda a descoberto (3) e 
    # pontuações ESG mínimas (4):
    cons = [cp.sum(allocations)==1, allocations<=0.10, 
            allocations>=0, E>=E_scr, S>=S_scr, G>=G_scr]
    
    #Função objetivo de média variância de Markowitz:
    obj = cp.Minimize(var - A*ret.values@allocations)
    
    #Otimização da variável de alocação dadas as restrições e a função acima:
    prob = cp.Problem(obj, cons)
    prob.solve()
    
    wgts = np.array(allocations.value.round(3))  

    #Colocando pesos e métricas em um Pandas DataFrame com o respectiva ação:    
    allocations_df = pd.DataFrame(wgts, index=data.index, columns=['Allocations'])
    allocations_df['Sectors'] = data.Sectors    
    allocations_df['Returns'] = ret
    
    #Colocando as pontuações ESG e variância em um Pandas Dataframe:
    port_metrics = {'Var': var.value, 'E_scr':E.value, 'S_scr':S.value, 
                    'G_scr':G.value, 'ESG_scr':esg.value}
    
    return allocations_df, port_metrics

# Função para criar os portfólios com as alocações e métricas:
def get_portfolio(E_scr, S_scr, G_scr, A, sectors=None, stocks=None):
    '''
    Retornos
    -------
    Dicionário com métricas de portfólio de risco, retorno e ESG. 
    2 gráficos de donut do Plotly, 1 para alocações de ações e 1 para alocações de setores.    
    '''
    #Alocações do Portfolio pela função de média variância:
    allocations, port_metrics = allocate_capital(E_scr, S_scr, G_scr, A, sectors, stocks)
    
    #Filtrar ações sem alocação:
    allocations = allocations[allocations['Allocations'] > 0]    

    #Dicionário das métricas de portfólio:
    metrics = {}

    #Pontuações ESG no dicionário:
    for scr in list(port_metrics.keys())[1:]:
        metrics[scr] = int(round(port_metrics[scr], 0))
        
    #Métricas de portfólio: Retorno e Vol Anuais
    metrics['Ret'] = (allocations.Returns@allocations.Allocations * 252 * 100).round(1)   
    metrics['Vol'] = (np.sqrt(port_metrics['Var']) * np.sqrt(252) * 100).round(1)         
    
    #Alocações em percentual:
    allocations.loc[:, 'Allocations'] = (allocations['Allocations'] * 100).round(1)

    #Alocações por setor:
    sector_allocs = allocations.groupby('Sectors').Allocations.sum()
    
    colors = ['#2E8B57', '#3CB371', '#66CDAA', '#8FBC8F', '#20B2AA', '#5F9EA0']
    
    #Gráfico em donut com alocações das ações:
    fig_stocks = go.Figure(data=[go.Pie(labels=allocations.index,
                                        values=allocations.Allocations.values,
                                        hole=0.35, pull=0.08,
                                        hoverinfo='label + percent')])

    fig_stocks.update_traces(marker=dict(colors=colors, 
                                         line=dict(color='#000000', width=1)),
                                         textposition='auto', 
                                         textfont_size=14,
                                         textinfo='percent+label',
                                         sort=False, 
                                         direction='clockwise', 
                                         rotation=90
                                         )

    fig_stocks.update_layout(font_color="black",
                             legend=dict(font=dict(size=12), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                             autosize=False,
                             width=700,
                             height=500,
                             modebar_remove=['toImage', 'hoverClosestPie'],
                             paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)",
                             margin=dict(l=0,r=100,b=0,t=60))    

    #Gráfico em donut com alocações por setor:
    fig_sectors = go.Figure(data=[go.Pie(labels=sector_allocs.index,
                                        values=sector_allocs.values,
                                        hole=0.7,
                                        hoverinfo='label + percent')])

    fig_sectors.update_traces(marker=dict(colors=colors, 
                                         line=dict(color='#000000', width=1)),
                                         textposition='auto', 
                                         textfont_size=14,
                                         textinfo='percent+label',
                                         sort=False, 
                                         direction='clockwise', 
                                         rotation=90)

    fig_sectors.update_layout(font_color="black",
                              legend=dict(font=dict(size=12), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                              autosize=False,
                              width=600,
                              height=400,
                              modebar_remove=['toImage', 'hoverClosestPie'],
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0,r=0,b=0,t=60))

    return metrics, fig_stocks, fig_sectors, allocations

# Função principal:
def main():

    selected = option_menu(
        menu_title='Portfólio ESG - Black-Litterman com Previsões VAR',
        options=['Home', 'Previsões VAR', 'Portfólio ESG', 'Métricas'],
        icons=['house', 'bandaid', 'basket', 'cash-coin'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )

    if selected == 'Home':
        st.sidebar.title('Portfólio ESG Brasil')
        st.sidebar.markdown('Eduardo Jr.')
        st.sidebar.write('''
                            Exame de qualificação apresentado ao
                            Programa de Pós-graduação em
                            Informática Aplicada, da Universidade de
                            Fortaleza, como requisito parcial para
                            obtenção do título de Mestre em
                            Informática Aplicada.
                         ''')
        st.sidebar.divider()
        st.sidebar.markdown("***Este trabalho não é, em hipótese nenhuma, recomendação de investimento.***")
        st.header('PORTFÓLIO DE INVESTIMENTOS EM EMPRESAS SUSTENTÁVEIS: UMA ABORDAGEM QUANTITATIVA HÍBRIDA')
        st.subheader('Resumo')
        with st.container(border=True):
            st.markdown(open('resumo.md').read())
        st.subheader('Definição dos portfólios e estratégia de comparação')
        with st.container(border=True):
            st.markdown(open('definicao.md').read())
            st.image("fig1.jpeg")
            st.image("fig2.jpeg")
        st.subheader('Conclusão')
        with st.container(border=True):
            st.markdown(open('conclusao.md').read())
        st.subheader('Ferramentas Utilizadas')
        with st.container(border=True):
            st.markdown(open('ferramentas.md').read())
        st.subheader('Referências')
        with st.container(border=True):
            st.markdown(open('referencias.md').read())

    elif selected == 'Previsões VAR':
        with st.sidebar:
            with st.sidebar.form('VAR'):
                st.title('Vetores Autorregressivos por Setor')        

                sector = st.selectbox('Selecione o Setor', ('Bens de Consumo Duráveis e Não Duráveis',
                                                            'Comércio e Tecnologia',
                                                            'Comunicações',
                                                            'Distribuição',
                                                            'Energia',  
                                                            'Financeiro',                                                            
                                                            'Imobiliário',
                                                            'Indústria de Processamento',
                                                            'Manufatura',
                                                            'Minerais',
                                                            'Serviços ao Consumidor',
                                                            'Serviços Básicos',
                                                            'Serviços de Saúde',                                                
                                                            'Serviços Industriais',
                                                            'Transportes',
                                                            'Varejista'))
                
                botao_form_var = st.form_submit_button('Obter Previsões VAR')

            with st.expander("Explicação"):
                st.write('''
                    - Modelo VAR aprende com retornos de 2019 a 2021 (2 anos):
                        - Início: 02-01-2019
                        - Fim: 04-01-2021
                    - Em seguida, Modelo VAR faz as previsões de retornos de 2021 a 2023 (2 anos):
                        - Início: 05-01-2021
                        - Fim: 05-01-2023
                    - Os retornos previstos são usados como entrada do Modelo BL que faz a alocação de portfólio.                    
                    ''')            
        
        if sector == 'Comércio e Tecnologia':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_commercial_technology_services_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Comunicações':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_communications_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Bens de Consumo Duráveis e Não Duráveis':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_consumer_durables_non_durables_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Serviços ao Consumidor':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_consumer_services_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Distribuição':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_distribution_services_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Energia':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_energy_minerals_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Financeiro':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_finance_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Serviços de Saúde':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_health_services_technology_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Serviços Industriais':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_industrial_services_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Minerais':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_non-energy_minerals_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Indústria de Processamento':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_process_industries_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Manufatura':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_producer_manufacturing_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Imobiliário':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_real_estate_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Varejista':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_retail_trade_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Transportes':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_transportation_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)
        elif sector == 'Serviços Básicos':
            df1 = pd.read_excel("./prices/adjusted_close/esgb11_utilities_prices.xlsx", engine="openpyxl")
            df1.set_index("Date", inplace=True)

        try:            
            if botao_form_var:
                    #Log returns
                    df = np.log(df1) - np.log(df1.shift(1))
                    df.dropna(inplace=True)
                    st.subheader(f"Retornos Logarítmicos - Ações do Setor {sector}")
                    st.dataframe(df)

                    #Correlação Heatmap
                    corr24 = df[0:496].corr()        
                    st.subheader('Correlação - 24 meses')
                    fig1 = px.imshow(corr24, text_auto=True, aspect='auto')
                    st.plotly_chart(fig1, theme='streamlit')

                    #Correlação Mediana Barras    
                    st.subheader('Correlação mediana da empresa com as demais')
                    empresas = corr24.median().sort_values(ascending=False)
                    fig2 = px.bar(x=empresas, y=empresas.index, orientation='h', text=round(empresas, 2))    
                    st.plotly_chart(fig2, theme='streamlit')

                    #Visualizando as séries dos retornos
                    st.subheader('Séries dos retornos')
                    sns.set_style("whitegrid")
                    num_cols = len(df.columns)
                    ncols = 2
                    nrows = math.ceil(num_cols / ncols)

                    fig = plt.figure(figsize=(16, nrows * 4), dpi=150)
                    fig.suptitle("Retornos Logarítmicos", fontsize=18, fontweight='bold', y=1.02)

                    for i, col in enumerate(df.columns):
                        ax = fig.add_subplot(nrows, ncols, i + 1)
                        df[col].plot(ax=ax, linewidth=2, color='tab:red')
                        ax.set_title(f"{col}", fontsize=12, fontweight='bold')                        
                        ax.tick_params(labelsize=8)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.set_xlabel("")
                        ax.set_ylabel("")                    
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Causalidade de Granger
                    st.subheader('Causalidade de Granger')
                    st.dataframe(grangers_causation_matrix(df, variables = df.columns))

                    # Teste de cointegração 
                    st.subheader('Teste de cointegração')
                    cointegration_test(df)

                    #Teste ADF
                    st.subheader('Teste ADF')
                    for name, column in df.items():
                        adfuller_test(column, name=column.name)

                    #Previsão VAR
                    st.subheader('Previsão de retornos utilizando Vetores Autoregressivos')
                    df_results = var_forecast(df)
                    st.dataframe(df_results)    

                    # Retornos - Atuais vs Previstos 1
                    st.subheader('Retornos Atuais vs Previstos - Série Completa')                    
                    sns.set_style("whitegrid")
                    num_cols = len(df.columns)
                    ncols = 2
                    nrows = math.ceil(num_cols / ncols)

                    fig = plt.figure(figsize=(16, nrows * 4), dpi=150)                    

                    for i, col in enumerate(df.columns):
                        ax = fig.add_subplot(nrows, ncols, i + 1)
                        df[col][496:996].plot(ax=ax, label="Real", linewidth=2, color='tab:blue')
                        df_results[col + '_previsto'].plot(ax=ax, label="Previsto", linewidth=2, color='tab:orange')
                        ax.set_title(f"{col}", fontsize=12, fontweight='bold')
                        ax.legend(fontsize=8, loc='upper left')
                        ax.tick_params(labelsize=8)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.set_xlabel("")
                        ax.set_ylabel("")          
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Retornos - Atuais vs Previstos 2
                    st.subheader('Retornos Atuais vs Previstos - Últimas 20 amostras')
                    sns.set_style("whitegrid")                    
                    num_cols = len(df.columns)
                    ncols = 2
                    nrows = math.ceil(num_cols / ncols)

                    fig = plt.figure(figsize=(16, nrows * 4), dpi=150)
                    fig.suptitle("Comparação: Valores Reais vs Previstos - Últimas 20 amostras", fontsize=18, fontweight='bold', y=1.02)

                    for i, col in enumerate(df.columns):
                        ax = fig.add_subplot(nrows, ncols, i + 1)
                        df[col][496:996].tail(20).plot(ax=ax, label="Real", linewidth=2, color='tab:blue')
                        df_results[col + '_previsto'].tail(20).plot(ax=ax, label="Previsto", linewidth=2, color='tab:orange')
                        ax.set_title(f"{col}", fontsize=12, fontweight='bold')
                        ax.legend(fontsize=8, loc='upper left')
                        ax.tick_params(labelsize=8)
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        ax.set_xlabel("")
                        ax.set_ylabel("")                    
                    plt.tight_layout()
                    st.pyplot(fig)               

                    # Exibir o DataFrame como CSV
                    csv = df_results.to_csv()
                    st.download_button(
                        label="Baixar CSV",
                        data=csv,
                        file_name=f"retornos_var_{sector.lower().replace(' ', '_')}.xlsx",
                        mime='text/csv'        
                    )
        except Exception as e:
            st.error(f"Error: {e}")

    elif selected == 'Portfólio ESG':
        with st.sidebar:
            with st.sidebar.form('Portfólio ESG Brasil'):

                st.title('Aspectos considerados em ESG e Risco')        
                st.write('1 - Questões ESG:')
                E_scr = st.slider('Qual nível de importância você classifica para os fatores ambientais?', 0, 100, 50)
                S_scr = st.slider('Qual nível de importância você classifica para os fatores sociais?', 0, 100, 50)
                G_scr = st.slider('Qual nível de importância você classifica para os fatores de governança?', 0, 100, 50)

                st.write('2 - Questão sobre Risco:')
                # Risk selection
                A = st.slider('Nível de Aversão a Risco:', 0.0, 5.0, 0.5)

                with st.expander("Explicação da Questão"):
                    st.markdown('''
                        Nível de risco em que se está disposto a evitar, sendo:
                        - 0: Disposto a alto risco, ou nível de aversão a risco baixo
                        - 5: Disposto a baixo risco, ou nível de aversão a risco alto
                    ''')    

                st.write('3 - Questões sobre Setores e Ações a evitar:')   

                # Stocks to avoid selection
                stocks = st_tags(
                    label='Ações a evitar:',
                    text='Pressione Enter para adicionar'
                )             

                # Sectors to avoid selection
                sectors = st.multiselect(
                    'Selecione os Setores a evitar:',
                    options=['Bens de Consumo Duráveis e Não Duráveis',
                             'Comércio e Tecnologia',
                             'Comunicações',
                             'Distribuição',
                             'Energia',
                             'Financeiro',                                                            
                             'Imobiliário',
                             'Indústria de Processamento',
                             'Manufatura',
                             'Minerais',
                             'Serviços ao Consumidor',
                             'Serviços Básicos',
                             'Serviços de Saúde',                                                
                             'Serviços Industriais',
                             'Transportes',
                             'Varejista']
                )

                botao_form = st.form_submit_button('Obter Portfólio')  

            with st.expander("Explicação da Estratégia"):
                st.write('''
                    - Os retornos previstos pelo Modelo VAR são as entradas do Modelo BL:
                        - Início: 05-01-2023 (Market Cap)
                    - O portfólio obtido e o benchmark são comparados entre 2023 e 2024.
                        - Início: 06-01-2023
                        - Fim: 01-08-2024
                    - As métricas financeiras são apresentadas.
                    ''')                
                
        if botao_form:        
            metrics, fig_stocks, fig_sectors, allocations = get_portfolio(int(E_scr), int(S_scr), int(G_scr), float(A), sectors, stocks)

            st.subheader('Pontuações e Risco')
            st.write(metrics)            
            st.subheader('Portfólio - Ações')
            st.plotly_chart(fig_stocks)
            st.subheader('Portfólio - Setores')
            st.plotly_chart(fig_sectors)

            st.subheader('Métricas do Portfólio')

            alloc = allocations[['Allocations', 'Sectors']].to_csv()

            allocations[['Allocations', 'Sectors']].to_csv('portfolio.csv')

            port = pd.read_csv('portfolio.csv')

            #risk_metrics(port)
            port.set_index("Ticker", inplace=True)
            p_col = port.index.values
            # Carregando dataframe com preços ajustados de todas as ações
            df_adj_prices = pd.read_excel("./stock_prices_adj.xlsx", engine="openpyxl")
            df_adj_prices.set_index("Date", inplace=True)
            p_prices = df_adj_prices[df_adj_prices.columns.intersection(p_col)]
            p_prices = p_prices[sorted(p_prices.columns)]
            p_prices = p_prices.sort_index(axis=0)
            p_pw = p_prices * (port['Allocations']/100)
            p_pw['Total'] = p_pw.iloc[:, 1:len(p_col)].sum(axis=1)
            # Carregando df com retornos VAR
            df_var_ret = pd.read_excel("./stock_prices_var.xlsx", engine="openpyxl")
            # Fazendo todos os ajustes anteriores para o df de retornos VAR
            df_var_ret.set_index("Date", inplace=True)
            p_var_ret = df_var_ret[df_var_ret.columns.intersection(p_col)]
            p_var_ret = p_var_ret[sorted(p_var_ret.columns)]
            p_var_ret = p_var_ret.sort_index(axis=0)
            # Alocação para retornos VAR deve ter pesos iguais
            p_var_ret = p_var_ret * (len(p_col)/100)
            p_var_ret['Total'] = p_var_ret.iloc[:, 1:len(p_col)].sum(axis=1)
            p_var_ret = p_var_ret.iloc[1:]
            benchmark = df_adj_prices['BOVA11']
            returns = p_pw.pct_change(1).dropna()
            #DF com retornos do portfolio, BOVA11 e VAR
            benchmark_r = benchmark.pct_change(1).dropna()
            portfolio_r = returns['Total']
            var_r = p_var_ret['Total']
            p_compare_returns = pd.concat([portfolio_r, benchmark_r, var_r], axis=1)
            p_compare_returns.columns = ['Portfolio BL Returns', 'Benchmark Returns', 'VAR Returns']
            # Resolvendo bug na lib pyfolio e pandas
            pd.DataFrame.iteritems = pd.DataFrame.items
            pd.Series.iteritems = pd.Series.items

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.write('Métricas de risco e retorno do Benchmark')
                table = pf.show_perf_stats(p_compare_returns['Benchmark Returns'])
                table = table.astype(str)
                st.table(table)
            with col2:
                st.write('Métricas de risco e retorno do Portfólio')
                table = pf.show_perf_stats(p_compare_returns['Portfolio BL Returns'])
                table = table.astype(str)
                st.table(table)
            with col3:
                st.write('Métricas de risco e retorno do VAR')
                table = pf.show_perf_stats(p_compare_returns['VAR Returns'])
                table = table.astype(str)
                st.table(table)
            
            st.download_button(
                label="Baixar CSV",
                data=alloc,
                file_name="./portfolio_esg_brasil.csv",
                mime='text/csv'        
                )
            
    elif selected == 'Métricas':
        with st.sidebar:
            with st.sidebar.form('Métricas'):
                st.title('Métricas Financeiras e de Risco dos Portfólios ESG')        

                portfolio = st.selectbox('Selecione o Portfólio', ('Portfolio 1',
                                                                   'Portfolio 2',
                                                                   'Portfolio 3',
                                                                   'Portfolio 4',
                                                                   'Portfolio 5',
                                                                ))
                
                botao_form_met = st.form_submit_button('Obter Métricas')

            with st.expander("Explicação"):
                st.write('''
                    - Os Portfólios ESG foram criados baseados nos seguintes critério ESG e de Risco:
                        - Portfólio 1: Muita ênfase aos critérios ESG com alto risco
                        - Portfólio 2: Muita ênfase aos critérios ESG com baixo risco
                        - Portfólio 3: Pouca ênfase aos critérios ESG com alto risco
                        - Portfólio 4: Pouca ênfase aos critérios ESG com baixo risco
                        - Portfólio 5: Ênfase média aos critérios ESG com risco médio
                    - Métricas Financeiras e de Risco além da comparação com Benchmark são apresentadas
                         - Início: 06-01-2023
                         - Fim: 01-08-2024
                    ''')

        if botao_form_met:
            if portfolio == 'Portfolio 1':
                port = pd.read_csv("./portfolios/portfolio_1.csv")
                st.header('Métricas do Portfólio 1 - Muita ênfase aos critérios ESG com alto risco')
                st.write('''
                            - Critério Environment: 86 de 100
                            - Critério Social: 85 de 100
                            - Critério Governance: 79 de 100
                            - Aversão a Risco: 1 de 5
                            - ESG Score: 84 de 100
                         ''')
            elif portfolio == 'Portfolio 2':
                port = pd.read_csv("./portfolios/portfolio_2.csv")
                st.header('Métricas do Portfólio 2 - Muita ênfase aos critérios ESG com baixo risco')
                st.write('''
                            - Critério Environment: 86 de 100
                            - Critério Social: 85 de 100
                            - Critério Governance: 79 de 100
                            - Aversão a Risco: 5 de 5
                            - ESG Score: 84 de 100
                         ''')
            elif portfolio == 'Portfolio 3':
                port = pd.read_csv("./portfolios/portfolio_3.csv")
                st.header('Métricas do Portfólio 3 - Pouca ênfase aos critérios ESG com alto risco')
                st.write('''
                            - Critério Environment: 40 de 100
                            - Critério Social: 47 de 100
                            - Critério Governance: 41 de 100
                            - Aversão a Risco: 1 de 5
                            - ESG Score: 42 de 100
                         ''')
            elif portfolio == 'Portfolio 4':
                port = pd.read_csv("./portfolios/portfolio_4.csv")
                st.header('Métricas do Portfólio 4 - Pouca ênfase aos critérios ESG com baixo risco')
                st.write('''
                            - Critério Environment: 45 de 100
                            - Critério Social: 53 de 100
                            - Critério Governance: 43 de 100
                            - Aversão a Risco: 5 de 5
                            - ESG Score: 47 de 100
                         ''')
            elif portfolio == 'Portfolio 5':
                port = pd.read_csv("./portfolios/portfolio_5.csv")
                st.header('Métricas do Portfólio 5 - Ênfase média aos critérios ESG com risco médio')
                st.write('''
                            - Critério Environment: 50 de 100
                            - Critério Social: 60 de 100
                            - Critério Governance: 50 de 100
                            - Aversão Risco: 2.5 de 5
                            - ESG Score: 54 de 100
                         ''')

            port.set_index("Ticker", inplace=True)
            p_col = port.index.values
            # Carregando dataframe com preços ajustados de todas as ações
            df_adj_prices = pd.read_excel("./stock_prices_adj.xlsx", engine="openpyxl")
            df_adj_prices.set_index("Date", inplace=True)
            p_prices = df_adj_prices[df_adj_prices.columns.intersection(p_col)]
            p_prices = p_prices[sorted(p_prices.columns)]
            p_prices = p_prices.sort_index(axis=0)
            p_pw = p_prices * (port['Allocations']/100)
            p_pw['Total'] = p_pw.iloc[:, 1:len(p_col)].sum(axis=1)
            # Carregando df com retornos VAR
            df_var_ret = pd.read_excel("./stock_prices_var.xlsx", engine="openpyxl")
            # Fazendo todos os ajustes anteriores para o df de retornos VAR
            df_var_ret.set_index("Date", inplace=True)
            p_var_ret = df_var_ret[df_var_ret.columns.intersection(p_col)]
            p_var_ret = p_var_ret[sorted(p_var_ret.columns)]
            p_var_ret = p_var_ret.sort_index(axis=0)
            # Alocação para retornos VAR deve ter pesos iguais
            p_var_ret = p_var_ret * (len(p_col)/100)
            p_var_ret['Total'] = p_var_ret.iloc[:, 1:len(p_col)].sum(axis=1)
            p_var_ret = p_var_ret.iloc[1:]
            benchmark = df_adj_prices['BOVA11']
            returns = p_pw.pct_change(1).dropna()
            #DF com retornos do portfolio, BOVA11 e VAR
            benchmark_r = benchmark.pct_change(1).dropna()
            portfolio_r = returns['Total']
            var_r = p_var_ret['Total']
            p_compare_returns = pd.concat([portfolio_r, benchmark_r, var_r], axis=1)
            p_compare_returns.columns = ['Portfolio BL Returns', 'Benchmark Returns', 'VAR Returns']
            # Resolvendo bug na lib pyfolio e pandas
            pd.DataFrame.iteritems = pd.DataFrame.items
            pd.Series.iteritems = pd.Series.items

            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.write('Métricas de risco e retorno do Benchmark') 
                table =  pf.show_perf_stats(p_compare_returns['Benchmark Returns'])
                table = table.astype(str)
                st.table(table)                
                plt.close('all')
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    pf.create_full_tear_sheet(p_compare_returns['Benchmark Returns'])
                fig_nums = plt.get_fignums()
                if not fig_nums:
                    st.warning("Nenhuma figura gerada pelo pyfolio")
                else:
                    for fig_num in fig_nums:
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)                
                plt.close('all')
            with col2:
                st.write('Métricas de risco e retorno do Portfólio')
                table = pf.show_perf_stats(p_compare_returns['Portfolio BL Returns'])
                table = table.astype(str)
                st.table(table)                
                plt.close('all')
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    pf.create_full_tear_sheet(p_compare_returns['Portfolio BL Returns'], benchmark_rets=p_compare_returns['Benchmark Returns'])
                fig_nums = plt.get_fignums()
                if not fig_nums:
                    st.warning("Nenhuma figura gerada pelo pyfolio")
                else:
                    for fig_num in fig_nums:
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)                
                plt.close('all')                
            with col3:
                st.write('Métricas de risco e retorno do VAR')
                table = pf.show_perf_stats(p_compare_returns['VAR Returns'])
                table = table.astype(str)
                st.table(table)                
                plt.close('all')
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    pf.create_full_tear_sheet(p_compare_returns['VAR Returns'], benchmark_rets=p_compare_returns['Benchmark Returns'])
                fig_nums = plt.get_fignums()
                if not fig_nums:
                    st.warning("Nenhuma figura gerada pelo pyfolio")
                else:
                    for fig_num in fig_nums:
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)                
                plt.close('all')                

if __name__ == '__main__':
    main()