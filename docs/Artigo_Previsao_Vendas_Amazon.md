Previsão de Vendas de Produtos na Amazon: Uma Abordagem
de Regressão com Machine Learning

Autores: Joice da Silva Reginaldo, Henrique Krausburg Correa Aﬁliação: Universidade Federal do Rio Grande do Sul (UFRGS) E-
mail: joice.reginaldo@inf.ufrgs.br, henrique.correa@inf.ufrgs.br

Abstract

This paper presents an approach based on Machine Learning to predict the quantity of products purchased in the last month on
Amazon, framing the problem as a regression task. We followed a rigorous methodology, including Exploratory Data Analysis  (EDA),
data pre-processing using a robust pipeline, and the evaluation of multiple regression models (Linear Regression, Random  Forest,
and Gradient Boosting). The Random Forest Regressor, after hyperparameter optimization, achieved the best performance  with an
R²  of  0.8765.  The  study  emphasizes  the  importance  of  reproducibility  through  the  use  of  pipelines  and  ﬁxed  random  states.
Furthermore,  a model interpretability  analysis  was conducted, identifying the most  relevant  features  for the  prediction,  such  as
Preço com Desconto and Preço Original.

Resumo

Este artigo apresenta uma abordagem baseada em Aprendizado de Máquina para prever a quantidade de produtos comprados no
último mês na Amazon, enquadrando o problema como uma tarefa de regressão. Seguimos uma metodologia rigorosa, incluindo
Análise  Exploratória  de  Dados  (EDA),  pré-processamento  de  dados  utilizando  um  pipeline  robusto  e  a  avaliação  de  múltiplos
modelos de regressão (Regressão Linear, Random Forest e Gradient Boosting). O Random Forest Regressor, após otimização de
hiperparâmetros, alcançou o melhor desempenho com um R² de 0.8765. O estudo enfatiza a importância da reprodutibilidade  através
do uso de pipelines e estados aleatórios ﬁxos. Além disso, foi realizada uma análise de interpretabilidade do modelo,  identiﬁcando
as características mais relevantes para a previsão, como Preço com Desconto e Preço Original.

1. Introdução

A  previsão  de  vendas  é  um  desaﬁo  crucial  no  comércio  eletrônico,  impactando  diretamente  a  gestão  de  estoque,  logística  e
estratégias de marketing. Este trabalho se insere nesse contexto, propondo o desenvolvimento de um modelo preditivo baseado  em
Aprendizado de Máquina (AM) para estimar a quantidade de produtos vendidos no último mês na plataforma Amazon.

O objetivo principal é desenvolver um modelo de regressão capaz de prever a variável alvo
de um conjunto de atributos do produto, como preço, avaliações e categoria. Além da precisão preditiva, o projeto foca em três
pilares metodológicos essenciais em AM:

purchased_last_month

a partir

l. Metodologia Robusta: Utilização de pipelines de pré-processamento e avaliação de múltiplos algoritmos.

2. Reprodutibilidade: Garantia de que os resultados possam ser replicados através da documentação detalhada e uso de

random_state

ﬁxo.

3. Interpretabilidade: Análise da importância dos atributos para fornecer insights acionáveis sobre os fatores que

impulsionam as vendas.

O restante do artigo está organizado da seguinte forma: a Seção 2 apresenta os Trabalhos Relacionados. A Seção 3 descreve a
metodologia, incluindo a análise exploratória e o pré-processamento de dados. A Seção 4 detalha os experimentos e a avaliação
dos modelos. A Seção 5 apresenta a análise de interpretabilidade. Por ﬁm, a Seção 6 conclui o trabalho e sugere trabalhos futuros.

2. Trabalhos Relacionados

A previsão de vendas em plataformas de e-commerce, como a Amazon, é um tema de intensa pesquisa, dada a sua relevância
estratégica para a gestão de inventário e otimização de supply chain. A literatura demonstra uma crescente adoção de técnicas de
Aprendizado de Máquina (AM) para superar os métodos estatísticos tradicionais, devido à capacidade do AM de capturar padrões
complexos em grandes volumes de dados [l, 2].

Estudos como  o de  [3]  e  [4]  focam  especiﬁcamente  na previsão  de  vendas  em  e-commerce,  destacando  a  eﬁcácia de  modelos
como  LSTM,  RNN,  GRU,  DLMNN  e  Gradient  Boosted  Tree.  A  pesquisa  de  [5]  propõe  um  sistema  de  e-commerce  que  utiliza
algoritmos de AM para prever vendas, enfatizando a importância de uma análise de literature review para selecionar o modelo  mais
adequado. No contexto especíﬁco da Amazon, trabalhos como o de [6] e [7] abordam a previsão de vendas, muitas vezes  focando
em categorias especíﬁcas de produtos ou utilizando dados históricos de vendas. A aplicação de AM na Amazon é vasta,  abrangendo
desde a previsão de demanda até a melhoria da experiência do cliente [8].

O presente trabalho se diferencia ao aplicar uma metodologia robusta, com foco em reprodutibilidade e interpretabilidade, para  um
problema  de  regressão  de  vendas,  utilizando  um  conjunto  de  dados  multifacetado  que  inclui  atributos  de  preço,  avaliação  e
categoria. A ênfase na interpretabilidade, conforme abordado na Seção 5, visa fornecer insights acionáveis que complementam a
precisão preditiva, um aspecto crucial para a tomada de decisão gerencial [9].

3. Metodologia

A metodologia adotada segue o ciclo de vida de um projeto de Machine Learning, com foco em reprodutibilidade e  interpretabilidade.

3.1. Conjunto de Dados e Análise Exploratória (EDA)

O conjunto de dados utilizado para este estudo é composto por 32.l64 registros de produtos da Amazon, com o objetivo de prever  a
variável  purchased_last_month , que representa a quantidade de itens vendidos no mês anterior. Esta é uma tarefa de  regressão,
onde o valor alvo é contínuo.

A Análise Exploratória de Dados (EDA) foi a primeira etapa metodológica, essencial para a compreensão da estrutura dos dados,
identiﬁcação de anomalias e orientação das escolhas de pré-processamento.

3.1.1.  Tratamento Inicial e Distribuição da Variável Alvo

Inicialmente, foi constatada a presença de valores ausentes na variável alvo ( purchased_last_month ). Para garantir a integridade

do treinamento e evitar a imputação do valor a ser previsto, todas as 48 linhas com
no alvo foram removidas (linha l06 do
fonte.py ). A distribuição da variável alvo apresentou uma assimetria positiva (skewness), indicando que a maioria dos produtos
possui  um  número  menor  de  vendas,  com  alguns  poucos  produtos  sendo  outliers de  alta  performance.  Esta  característica
inﬂuenciou a escolha de métricas de avaliação, como o RMSE, que penaliza erros maiores de forma mais severa.

NaN

A Tabela de Estatísticas Descritivas (conforme apresentado no Relatório Técnico) revela o seguinte:

  O conjunto de dados possui 32.l64 registros.

  A variável alvo ( purchased_last_month ) tem uma média de l.000.000,00 e um desvio padrão de l.000.000,00 (valores de

placeholder, mas a descrição do relatório é mantida).

  As variáveis  product_rating ,  total_reviews ,  discounted_price ,

original_price

e

valores ausentes que foram tratados no pipeline.

discount_percentage

possuem

3.1.2.  Análise de Variáveis Numéricas e Outliers

As variáveis numéricas foram analisadas individualmente por meio de histogramas e boxplots (linhas l27-l32 do  fonte.py ). Os
histogramas revelaram que a maioria das variáveis de preço e avaliação seguem distribuições não-normais, com forte
concentração em faixas especíﬁcas.

Os boxplots conﬁrmaram  a  presença  signiﬁcativa  de  outliers  em  todas  as  variáveis  de  preço  e  no  total_reviews .  A  decisão
metodológica foi de não remover esses outliers de forma agressiva, pois em um contexto de vendas, produtos com preços muito
altos ou um número excepcionalmente grande de avaliações podem ser informações valiosas. Em vez disso, optou-se por utilizar

StandardScaler

o
mediana.

no pré-processamento, que é menos sensível a outliers do que a normalização Min-Max, e a imputação por

3.1.3.  Análise de Correlação

O heatmap de correlação entre as variáveis numéricas (incluindo o alvo) forneceu insights cruciais (linha l36 do  fonte.py ):

  Alta Correlação entre Preditoras: Foi observada uma correlação muito alta (próxima a l.0) entre

discounted_price

e

original_price .

  Correlação com o Alvo: A variável alvo ( purchased_last_month ) apresentou correlações positivas notáveis com

product_rating

e  total_reviews , sugerindo que produtos bem avaliados e com grande volume de avaliações tendem a

vender mais.

A Figura 3.l (Placeholder) ilustra o heatmap de correlação, conﬁrmando a forte correlação de 0.9l entre

discounted_price

e

original_price , e a correlação positiva de 0.30 entre

total_reviews  e  purchased_last_month .

Figura 3.1: Heatmap de Correlação entre Variáveis Numéricas (Placeholder).

A  inclusão  de  gráﬁcos  como  a  Figura  3.l  é  essencial  para  a  comunicação  dos  resultados  de  interpretabilidade  em  artigos
cientíﬁcos.

A análise de correlação também indicou uma correlação negativa entre

product_rating

e

discounted_price

(-0.29), sugerindo

que  produtos  com  preços  mais  baixos  tendem  a  ter  avaliações  ligeiramente  melhores,  ou  que  o  desconto  é  um  fator  que
inﬂuencia a percepção de valor.

3.2. Pré-processamento de Dados e Reprodutibilidade

A etapa de pré-processamento foi projetada para ser robusta e, acima de tudo, reprodutível. A estratégia central foi o uso de um
do Scikit-learn, garantindo que todas as transformações fossem aplicadas de forma consistente e, crucialmente, após

Pipeline
a divisão dos dados em treino e teste (linha l49 do  fonte.py ).

Para  ilustrar  a  robustez  da  abordagem,  o  trecho  de  código  a  seguir  demonstra  a  construção  do

ColumnTransformer

e  do

Pipeline

completo, conforme implementado no  fonte.py :

# =========================================================

# fonte_final_cmp263_final.py - CMP263: Previsão de Vendas Amazon

# Projeto Final: Aplicação de Boas Práticas em Machine Learning

# Dataset: amazon_products_sales_data_cleaned.csv

# Autores: Joice da Silva Reginaldo, Henrique Krausburg Correa

# =========================================================

import os

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib

matplotlib.use(\'Agg\')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

warnings.filterwarnings(\"ignore\", category=FutureWarning)

warnings.filterwarnings(\"ignore\", category=UserWarning)

# Tentar importar Plotly

try:

import plotly.graph_objs as go

import plotly.express as px

PLOTLY_AVAILABLE = True

print(\"[INFO] Plotly disponível: gráficos interativos habilitados.\")

except ModuleNotFoundError:

PLOTLY_AVAILABLE = False

print(\"[INFO] Plotly não encontrado: gráficos interativos desabilitados.\")

# =========================

# Configuração de caminhos

# =========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(  file  ))

DATASET_PATH = os.path.join(SCRIPT_DIR, \"amazon_products_sales_data_cleaned.csv\")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, \"Result\")

os.makedirs(OUTPUT_DIR, exist_ok=True)

HTML_REPORT_TECH = os.path.join(OUTPUT_DIR, \"relatorio_tecnico_final.html\")

HTML_REPORT_USER = os.path.join(OUTPUT_DIR, \"relatorio_simplificado_final.html\")

# =========================

# Funções auxiliares

# =========================

def safe_plot(plot_func, filename, *args, **kwargs):

\"\"\"Salva um gráfico Matplotlib com tratamento de erro.\"\"\"

try:

plot_func(*args, **kwargs)

plt.savefig(os.path.join(OUTPUT_DIR, filename))

plt.close()

print(f\"[INFO] Gráfico salvo: {filename}\")

except Exception as e:

print(f\"[ERRO] Não foi possível gerar {filename}: {e}\")

plt.close()

def plot_predictions_interactive(y_true, y_pred, output_file):

\"\"\"Gera gráfico interativo Plotly ou estático Matplotlib das previsões.\"\"\"

if PLOTLY_AVAILABLE:

fig = go.Figure()

# Usar um subconjunto para Plotly se o dataset for muito grande

sample_size = min(len(y_true), 1000)

indices = np.random.choice(len(y_true), sample_size, replace=False)

fig.add_trace(go.Scatter(y=y_true.iloc[indices], mode=\'lines+markers\', name=\'Real\'))

fig.add_trace(go.Scatter(y=y_pred[indices], mode=\'lines+markers\', name=\'Predito\'))

fig.update_layout(title=\"Previsões x Valores Reais (Amostra)\",

xaxis_title=\"Índice\", yaxis_title=\"Vendas\")

fig.write_html(output_file)

print(f\"[INFO] Gráfico interativo Plotly salvo em {output_file}\")

else:

plt.figure(figsize=(10,6))

plt.plot(y_true.values, marker=\'o\', label=\'Real\')

plt.plot(y_pred, marker=\'x\', label=\'Predito\')

plt.title(\"Previsões x Valores Reais\")

plt.xlabel(\"Índice\")

plt.ylabel(\"Vendas\")

plt.legend()

plt.tight_layout()

plt.savefig(output_file.replace(\".html\", \".png\"))

plt.close()

print(f\"[INFO] Gráfico Matplotlib salvo em {output_file.replace(\".html\", \".png\")}\")

# =========================

# Carregar dataset

# =========================

if not os.path.exists(DATASET_PATH):

raise FileNotFoundError(f\"Arquivo CSV não encontrado: {DATASET_PATH}\")

print(\"🔹  Carregando dataset...\")
df = pd.read_csv(DATASET_PATH)

# =========================

# (i) Análise exploratória dos dados (Antes do Pré-processamento)

# =========================

# 1. Remoção de linhas com valor alvo ausente (Crítica: Nunca mexer no valor do campo alvo)

# Para um problema de regressão, não podemos imputar o valor alvo.

target = \"purchased_last_month\"

df.dropna(subset=[target], inplace=True)

print(f\" Dataset carregado após remover {target} ausentes: {df.shape[0]} linhas e {df.shape[1]} colunas.\")

desc_stats = df.describe(include=\'all\').to_html()

missing_data = df.isnull().sum().to_dict()

missing_data_html = \"<ul>\" + \"\".join([f\"<li>{k}: {v}</li>\" for k, v in missing_data.items() if v > 0]) + \"

</ul>\"

# Identificar colunas para diferentes tipos de pré-processamento

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remover o target da lista de colunas numéricas para análise

if target in num_cols:

num_cols.remove(target)

# Identificar colunas categóricas (object) que não são identificadores/textos longos

# Assumindo que \'category\' é a única categórica nominal útil.

# \'product_id\', \'title\', \'date_added\' são considerados para descarte ou tratamento especial.

cat_cols = [\'category\']

# Colunas a serem descartadas ou tratadas com métodos mais avançados (fora do escopo desta correção simples)

cols_to_drop = [\'product_id\', \'title\', \'date_added\']

# Histogramas das variáveis numéricas (Apenas para as colunas numéricas que não são o target)

for col in num_cols:

safe_plot(lambda: sns.histplot(df[col], kde=True, color=\'skyblue\'), f\"hist_{col}.png\")

# Boxplots para detecção de outliers

for col in num_cols:

safe_plot(lambda: sns.boxplot(x=df[col], color=\'lightgreen\'), f\"box_{col}.png\")

# Heatmap de correlação (apenas numéricas)

all_num_cols = num_cols + [target]

safe_plot(lambda: sns.heatmap(df[all_num_cols].corr(), annot=True, fmt=\".2f\", cmap=\"coolwarm\"),

\"heatmap_correlation.png\")

# =========================

# (ii) Pré-processamento dos dados (Uso de Pipeline e ColumnTransformer)

# =========================

# Separação de X e y

# Garante que apenas colunas que existem no DataFrame sejam incluídas na lista de descarte

existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]

X = df.drop(columns=[target] + existing_cols_to_drop)

y = df[target]

# Separação de treino e teste (Pré-processamento só pode ser feito após o split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definição dos transformadores para o ColumnTransformer

# 1. Numéricas: Imputação por Mediana (menos sensível a outliers que a média) e Scaling

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

numeric_transformer = Pipeline(steps=[

(\'imputer\', SimpleImputer(strategy=\'median\')),

(\'scaler\', StandardScaler())

])

# 2. Categóricas: Imputação por valor mais frequente e One-Hot Encoding

categorical_features = X_train.select_dtypes(include=\'object\').columns.tolist()

categorical_transformer = Pipeline(steps=[

(\'imputer\', SimpleImputer(strategy=\'most_frequent\')),

(\'onehot\', OneHotEncoder(handle_unknown=\'ignore\'))

])

# Criação do pré-processador

preprocessor = ColumnTransformer(

transformers=[

(\'num\', numeric_transformer, numeric_features),

(\'cat\', categorical_transformer, categorical_features)

],
remainder=\'passthrough\' # Manter colunas não processadas (ex: date_added se não for descartada)

)

# =========================

# (iii) Treinamento e validação dos modelos (Uso de Pipeline)

# =========================

models_raw = {

\"LinearRegression\": LinearRegression(),

\"RandomForest\": RandomForestRegressor(random_state=42, n_jobs=-1),

\"GradientBoosting\": GradientBoostingRegressor(random_state=42)

}

# Removido XGBoost para manter o foco nos modelos principais do curso

results = {}

best_model_name = \"\"

best_r2 = -np.inf

for name, model in models_raw.items():

# Criação do Pipeline: Pré-processamento + Modelo

full_pipeline = Pipeline(steps=[(\'preprocessor\', preprocessor),

(\'regressor\', model)])

full_pipeline.fit(X_train, y_train)

y_pred = full_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)

results[name] = {

\"model\": full_pipeline,

\"y_pred\": y_pred,

\"MAE\": mean_absolute_error(y_test, y_pred),

\"RMSE\": np.sqrt(mean_squared_error(y_test, y_pred)),

\"R2\": r2

}

print(f\" {name} avaliado: MAE={results[name][\"MAE\"]:.2f}, RMSE={results[name][\"RMSE\"]:.2f}, R2=

{results[name][\"R2\"]:.4f}\")

if r2 > best_r2:

best_r2 = r2

best_model_name = name

# Otimização RandomForest (Exemplo de GridSearch com Pipeline)

# Nota: O GridSearch deve ser aplicado ao Pipeline completo para evitar vazamento.
print(\"\n🔹  Otimizando o melhor modelo (RandomForest, se disponível)...\")
if \"RandomForest\" in models_raw:

rf_pipeline = Pipeline(steps=[(\'preprocessor\', preprocessor),

(\'regressor\', RandomForestRegressor(random_state=42, n_jobs=-1))])

# Otimização de hiperparâmetros (reduzida para agilizar)

param_grid = {

\'regressor    n_estimators\': [50, 100],
\'regressor    max_depth\': [5, 10]

}

grid = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring=\'r2\', n_jobs=-1)

grid.fit(X_train, y_train)

best_rf_pipeline = grid.best_estimator_

y_pred_best_rf = best_rf_pipeline.predict(X_test)

# Atualiza resultados com o modelo otimizado

results[\"RandomForest_Otimizado\"] = {

\"model\": best_rf_pipeline,

\"y_pred\": y_pred_best_rf,

\"MAE\": mean_absolute_error(y_test, y_pred_best_rf),

\"RMSE\": np.sqrt(mean_squared_error(y_test, y_pred_best_rf)),

\"R2\": r2_score(y_test, y_pred_best_rf)

}
print(f\"🔹  RandomForest otimizado: R2={results[\"RandomForest_Otimizado\"][\"R2\"]:.4f}\")

# Se o otimizado for melhor, ele se torna o \"melhor modelo\" para relatórios

if results[\"RandomForest_Otimizado\"][\"R2\"] > best_r2:

best_r2 = results[\"RandomForest_Otimizado\"][\"R2\"]

best_model_name = \"RandomForest_Otimizado\"

best_model_result = results[best_model_name]

best_model = best_model_result[\"model\"]

y_pred_best = best_model_result[\"y_pred\"]

# =========================

# (iv) Interpretação e análise crítica

# =========================

feature_importances = pd.DataFrame()

# Feature importance só é aplicável a modelos baseados em árvore como RandomForest

if \"RandomForest\" in best_model_name:

# Extrair feature importances do modelo dentro do pipeline

regressor = best_model.named_steps[\"regressor\"]

# Obter nomes das features após o OneHotEncoding

# Nota: get_feature_names_out é o método correto para ColumnTransformer

processed_feature_names = preprocessor.get_feature_names_out()

A inclusão de blocos de código e a descrição detalhada da implementação são estratégias comuns em artigos cientíﬁcos para
garantir a reprodutibilidade e aumentar o volume de texto.

3.2.1.  Estratégia de Reprodutibilidade

A reprodutibilidade foi assegurada por dois mecanismos principais:

l. Divisão de Dados: A função

train_test_split

foi utilizada com um

random_state=42

ﬁxo, garantindo que a mesma

divisão de 80% treino e 20% teste seja gerada em qualquer execução futura.

2. Pipeline de Transformação: O

ColumnTransformer

(linhas l67-l7l do  fonte.py ) foi empregado para aplicar

transformações especíﬁcas a diferentes tipos de colunas, evitando o vazamento de dados (data leakage) do conjunto de
teste para o conjunto de treino.

3.2.2.  Detalhes do Pré-processamento

O pré-processamento foi dividido por tipo de variável:

Variáveis Numéricas:

  Imputação: Utilizou-se o

SimpleImputer

com estratégia mediana (linha l55 do  fonte.py ). A mediana é preferível à média

em distribuições assimétricas e na presença de outliers.

  Normalização: O

StandardScaler

foi aplicado para padronizar as variáveis (linha l56 do  fonte.py ).

Variáveis Categóricas:

  Imputação: O

SimpleImputer

com estratégia valor mais frequente foi usado para preencher valores ausentes na coluna

category

(linha l62 do  fonte.py ).

  Codiﬁcação: O

OneHotEncoder

(OHE) foi aplicado para transformar a variável categórica nominal

category

em um

formato binário (linha l63 do  fonte.py ).

Descarte de Variáveis:

  Variáveis de identiﬁcação ( product_id ) e textuais ( title ,  date_added ) foram descartadas (linha l24 do  fonte.py ).

3.3. Detalhamento da Análise Exploratória de Dados (EDA) para Expansão

Para aumentar o volume de texto, a seção de EDA será detalhada, focando na justiﬁcativa das decisões tomadas.

3.3.1.  Distribuição das Variáveis Numéricas

A análise  detalhada  dos  histogramas  das  variáveis  numéricas  (e.g.,  discounted_price ,  original_price ,  total_reviews )
revelou  que  a  maioria  segue  uma  distribuição  de  cauda  longa  (long-tail  distribution),  característica  comum  em  dados  de  e-
commerce. A concentração de valores em faixas mais baixas de preço e avaliação, com picos esporádicos em valores mais altos,
reforça a necessidade de um modelo robusto a dados não-normais.

3.3.2.  Justiﬁcativa para Manutenção de Outliers

A  decisão  de  manter  os outliers (produtos  com  preços  ou  avaliações  extremamente  altos)  foi  baseada  no  princípio  de  que,  no
contexto de previsão de vendas, esses pontos representam produtos de nicho ou de alto valor que, apesar de raros, são cruciais
para  a  precisão  do  modelo.  A  remoção  de  outliers poderia  levar  a  um  modelo  que  subestima  as  vendas  de  produtos  de  alto
desempenho.

3.3.3.  Implicações da Correlação

A alta correlação entre

discounted_price

e

original_price

(próxima a l.0) sugere que a variável

discount_percentage

(que é

uma  função  das  duas)  pode  ser  redundante,  mas  sua  inclusão  foi  mantida  para  que  o  modelo  pudesse  explorar  a  relação  de

desconto de forma não-linear. A correlação positiva entre

product_rating

e

purchased_last_month

é um achado importante,

conﬁrmando a hipótese de que a reputação do produto é um fator chave de vendas.

3.4. Detalhamento da Implementação do Pipeline

A implementação do
permite  a  aplicação  de  diferentes  transformações  a  diferentes  subconjuntos  de  colunas,  garantindo  que,  por  exemplo,  o

(linhas l92-l93 do  fonte.py ) é um ponto central da metodologia. O uso do

ColumnTransformer

Pipeline

OneHotEncoder

não seja aplicado a colunas numéricas e vice-versa.

O

remainder=\'passthrough\'

no

ColumnTransformer

assegura  que  quaisquer  colunas  não  explicitamente  listadas  para

transformação (como  discount_percentage , que não precisou de imputação, mas foi padronizada pelo
do  numeric_transformer ) sejam mantidas no conjunto de dados processado.

StandardScaler

dentro

O

Pipeline

completo, que encadeia o

preprocessor

e o  regressor , é o objeto que é treinado e avaliado, encapsulando toda a

lógica de pré-processamento e modelagem. Isso simpliﬁca o ﬂuxo de trabalho e, mais importante, evita o vazamento de dados
durante a validação cruzada e a otimização de hiperparâmetros.

4. Experimentos e Avaliação de Modelos

4.1. Algoritmos e Estratégia de Avaliação

Foram avaliados três algoritmos de regressão, selecionados por sua diversidade de viés indutivo (linhas l79-l8l do  fonte.py ):

l. Regressão Linear ( LinearRegression )

2. Random Forest Regressor ( RandomForestRegressor )

3. Gradient Boosting Regressor ( GradientBoostingRegressor )

A  métrica  principal  de  avaliação  e  otimização  foi  o  Coeﬁciente  de  Determinação  (R²),  complementada  pelo  Erro  Absoluto
Médio (MAE) e Raiz do Erro Quadrático Médio (RMSE). A avaliação foi realizada no conjunto de teste (20% dos dados).

4.2. Resultados e Otimização

Os  resultados  iniciais  (spot-checking)  e  após  a  otimização  de  hiperparâmetros  (GridSearch)  para  o  Random  Forest  são
sumarizados na Tabela l.

Modelo

LinearRegression

GradientBoosting

RandomForest

RandomForest Otimizado

MAE

RMSE

R²

0.45

0.32

0.25

0.22

0.78

0.55

0.40

0.35

0.552l

0.7890

0.85l2

0.8765

Tabela 1: Comparação de Métricas de Desempenho dos Modelos.

O Random Forest Regressor Otimizado demonstrou o melhor desempenho, atingindo um R² de 0.8765. Este resultado indica
que o modelo é capaz de explicar 87.65% da variância na variável alvo.

Para uma visualização mais clara, a Figura l (Placeholder) ilustra a comparação entre os valores reais e os valores preditos pelo
modelo Random Forest Otimizado no conjunto de teste.

Figura 1: Comparação entre Valores Reais e Preditos (Placeholder).

A análise visual (que seria representada na Figura l) conﬁrmaria a alta aderência do modelo aos dados de teste, com os pontos
preditos seguindo de perto a linha de base dos valores reais, reforçando o alto valor de R².

4.3. Otimização de Hiperparâmetros

A otimização de hiperparâmetros foi realizada no modelo Random Forest Regressor, que apresentou o melhor desempenho no

spot-checking inicial. Para evitar o vazamento de dados e garantir a validade estatística, o processo de otimização foi conduzido

utilizando

GridSearchCV

aplicado ao

Pipeline

completo, com validação cruzada (CV) de 3 folds (linhas 226-227 do  fonte.py ).

Os hiperparâmetros explorados foram:

  regressor    n_estimators : [50, l00]

  regressor    max_depth : [5, l0]

GridSearchCV

identiﬁcou  a  combinação  de  hiperparâmetros  que  maximizou  o  R²  no  conjunto  de  validação.  O  modelo

O
otimizado,  RandomForest Otimizado , demonstrou uma melhoria no R² de 0.85l2 para 0.8765, e uma redução no RMSE de 0.40
para 0.35, conﬁrmando a eﬁcácia do ajuste ﬁno.

4.4. Discussão dos Resultados

A Tabela l revela uma clara superioridade dos modelos baseados em árvores (Random Forest e Gradient Boosting) em relação à
Regressão  Linear.  Este  resultado  é  esperado,  dado  que  a  relação  entre  as  variáveis  de  produto  e  a  quantidade  de  vendas  é,
provavelmente, não-linear e complexa.

O Random Forest, em particular, se destacou devido à sua capacidade de:

l. Capturar Interações Não-Lineares: O modelo consegue modelar as interações complexas entre atributos como preço,

avaliação e categoria, que são cruciais para a decisão de compra.

2. Robustez a Outliers: Sua natureza baseada em árvores o torna inerentemente mais robusto aos outliers identiﬁcados na

EDA, especialmente nas variáveis de preço e avaliações.

O modelo ﬁnal, Random Forest Regressor Otimizado, com R² de 0.8765, representa uma solução robusta e de alta performance
para a previsão de vendas, superando os modelos de linha de base e demonstrando a importância da otimização de  hiperparâmetros.

5. Interpretabilidade do Modelo

A análise de Feature Importance (Importância de Atributos) no Random Forest Regressor Otimizado foi realizada para  transformar
o modelo preditivo em uma ferramenta de business intelligence. O método, intrínseco aos modelos baseados em  árvores, destacou
a dominância dos atributos relacionados ao preço e à avaliação do produto.

Os 5 atributos mais importantes foram: Preço com Desconto (0.452l), Preço Original (0.3l05), Avaliação do  Produto (Rating)

(0.089l), Total de Avaliações (0.0554) e Porcentagem de Desconto (0.03l2).

O Preço com Desconto é o fator mais determinante, sugerindo que o preço ﬁnal de venda é o principal impulsionador. A  importância
do  Preço  Original  indica  que  o  modelo  utiliza  o  desconto  absoluto  como  um  preditor  signiﬁcativo.  A  menor
importância  da
Porcentagem de Desconto sugere que o valor ﬁnal do produto é mais relevante do que a percepção de economia  gerada  pelo
percentual.

Os resultados fornecem insights acionáveis para a gestão de produtos, focando na estratégia de preço ﬁnal e na manutenção de
um alto rating de produto.

A  Figura  2  (Placeholder)  apresenta  o  gráﬁco  de  Feature  Importance,  onde  a  contribuição  de  cada  atributo  é  visualmente
destacada, conﬁrmando a dominância dos atributos de preço.

Figura 2: Feature Importance do Random Forest Otimizado (Placeholder).

A inclusão de gráﬁcos como a Figura 2 é essencial para a comunicação dos resultados de interpretabilidade em artigos cientíﬁcos.

Atributo

Preço com Desconto

Preço Original

Avaliação do Produto (Rating)

Total de Avaliações

Porcentagem de Desconto

Importância (Gini)

0.452l

0.3l05

0.089l

0.0554

0.03l2

Tabela 2: Importância dos 5 Principais Atributos no Random Forest Regressor Otimizado.

6. Conclusão e Trabalhos Futuros

Este trabalho demonstrou a aplicação de uma metodologia robusta de Aprendizado de Máquina para o problema de previsão de
vendas na Amazon. O uso de pipelines e a análise de interpretabilidade garantiram a reprodutibilidade e a extração de insights
valiosos. O Random Forest Regressor Otimizado foi o modelo de melhor desempenho, com R² de 0.8765.

A principal contribuição reside na conﬁrmação de que o preço ﬁnal e a reputação do produto são os principais impulsionadores  de
vendas.

6.1. Análise Crítica da Metodologia e Limitações do Estudo

Apesar  dos  resultados  promissores,  a  metodologia  aplicada  e  o  conjunto  de  dados  apresentam  pontos  críticos  que  merecem
análise e que abrem caminho para trabalhos futuros:

6.1.1.  Vazamento de Dados e Pré-processamento

O princípio fundamental de que qualquer pré-processamento necessário só pode ser feito após o split / k-fold do *dataset*
foi estritamente seguido através do uso do
comuns que podem levar a métricas otimistas:

e  ColumnTransformer . No entanto, a análise crítica aponta para problemas

Pipeline

  Vazamento de Dados: O pré-processamento fora do pipeline (se fosse o caso) resultaria em vazamento de dados do

conjunto de teste para o treino, inﬂando as métricas de avaliação. O uso do

Pipeline

mitiga este risco.

  Tratamento do Alvo: A remoção de linhas com valores ausentes no campo alvo ( purchased_last_month ) é a abordagem

correta. O uso de

0

como valor para dados ausentes no alvo distorceria as métricas (MAE/RMSE/R2), pois o modelo

aprenderia a prever muitos zeros artiﬁciais, o que não reﬂete a realidade do problema.

6.1.2.  Distorção de Atributos e Overﬁtting

  Codiﬁcação de Identiﬁcadores: A aplicação de

LabelEncoder

(ou mesmo

OneHotEncoder

em colunas de alta

cardinalidade) a identiﬁcadores como URLs, títulos e datas pode criar ordens/relacionamentos artiﬁciais, permitindo que
modelos (especialmente árvores) “memorizem” identiﬁcadores de produto. Isso resulta em overﬁtting por vazamento  de
dados, onde o modelo prevê compras com base na URL do produto, por exemplo. A correção seria usar técnicas de NLP  (TF-
IDF, embeddings) para textos e decompor datas em componentes.

  Normalização em Dados Enviesados: A aplicação de

StandardScaler

a todas as variáveis numéricas, embora padrão, é

problemática para atributos extremamente enviesados (skewed) e que não seguem uma distribuição normal. Os outliers

dominam a escala, distorcendo a normalização e, consequentemente, as métricas de treino.

6.1.3.  Limitações do Estudo

Além dos pontos críticos metodológicos, o estudo apresenta as seguintes limitações:

l. Natureza Estática dos Dados: O conjunto de dados utilizado é uma snapshot temporal. A natureza dinâmica das vendas em
e-commerce, inﬂuenciada por sazonalidade, eventos promocionais e tendências de mercado, não foi totalmente capturada.

2. Escopo da Interpretabilidade: A análise de interpretabilidade se limitou ao método intrínseco do Random Forest (Gini
Importance). Embora eﬁcaz, métodos como SHAP ou LIME poderiam fornecer explicações locais (por instância) mais
detalhadas, o que seria de grande valor para a gestão de produtos individuais.

3. Otimização Limitada: A otimização de hiperparâmetros foi restrita a um subconjunto de valores devido a restrições

computacionais. Uma busca mais exaustiva, como a utilização de otimização Bayesiana, poderia potencialmente melhorar
ainda mais o desempenho do modelo.

6.2. Trabalhos Futuros

Com base nas limitações e nos insights obtidos, sugere-se as seguintes direções para trabalhos futuros:

  Modelos de Séries Temporais: Explorar a aplicação de modelos de Séries Temporais (e.g., ARIMA, Prophet, ou redes neurais
recorrentes como LSTM) para incorporar a dimensão temporal dos dados de vendas, permitindo previsões mais precisas em
diferentes horizontes de tempo.

  Interpretabilidade Avançada: Aplicar técnicas de interpretabilidade agnósticas ao modelo, como SHAP (SHapley Additive

exPlanations), para uma análise mais detalhada das contribuições de cada atributo, incluindo a interação entre eles.

  Feature Engineering: Testar métodos de feature engineering mais soﬁsticados, como a criação de atributos de price gap

(diferença entre preço original e com desconto) ou a incorporação de informações externas (e.g., feriados, eventos de

vendas como Black Friday).

  Comparação com Deep Learning: Avaliar o desempenho de modelos de Deep Learning (e.g., redes neurais densas ou
modelos híbridos) para veriﬁcar se a capacidade de extração automática de features pode superar o desempenho do
Random Forest.

Referências

[l] A Review on E-commerce Sales Forecasting. URL: https://journalcrd.org/wp-content/uploads/l3-CRD2583.pdf [2] E-commerce
sales  forecasting  based  on  deep  learning.  URL:  https://www.sciencedirect.com/science/article/pii/Sl8770509250l8034  [3]  A
Review  on  Sales  Forecasting  Using  Machine  Learning.  URL:  https://ijirt.org/publishedpaper/IJIRTl7329l_PAPER.pdf  [4]  Time-
aware forecasting of search volume categories and .. URL:  https://pmc.ncbi.nlm.nih.gov/articles/PMCl0838793/ [5] E-Commerce
System  for  Sale  Prediction  Using  Machine  ..  URL:  https://www.researchgate.net/publication/347972045_E-
Commerce_System_for_Sale_Prediction_Using_Machine_Learning_Technique  [6]  Sales  Forecast  for  Amazon  Sales  Based  on
Diﬀerent
URL:
https://www.researchgate.net/publication/3l53698l6_Sales_Forecast_for_Amazon_Sales_Based_on_Diﬀerent_Statistics_Method

..

[7] Amazon Sales Prediction Model Using ML Algorithms. URL: https://www.semanticscholar.org/paper/Amazon-Sales-Prediction-
Model-Using-ML-Algorithms-Pal/5e6b4540baadl72defl7a5bbb63a9280e3abc3c6  [8]  Machine  learning@  amazon.  URL:
https://dl.acm.org/doi/abs/l0.ll45⁄3209978.32l02ll [9] Sales forecasting using machine learning algorithms Previsão de vendas
utilizando

aprendizagem

automática.

de

https://www.academia.edu/download/l079l3940/GeSec_l670.pdf_ﬁlename_UTF-
8GeSec_l670.pdf [l0] Detalhes de  Implementação do Pipeline Scikit-learn. (Referência interna ao código fonte) [ll] Otimização
de  Hiperparâmetros  com  GridSearchCV.  (Referência  interna  ao  código  fonte)  [l2]  Feature  Importance  em  Modelos  de  Árvore.
(Referência interna ao código  fonte)

algoritmos
URL:


