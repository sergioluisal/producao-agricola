import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Análise Agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título do uploader
st.markdown('<h1 class="main-header">🌾 Análise Agrícola</h1>', unsafe_allow_html=True)

# Upload de arquivo
uploaded_file = st.file_uploader("📂 Carregue seu arquivo de dados (CSV, XLSX)", type=["csv", "xls", "xlsx"])

# Função para carregar dados
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame()

    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "csv":
            encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                    break
                except Exception:
                    uploaded_file.seek(0)
                    continue
            if df is None:
                st.error("Não foi possível decodificar o arquivo CSV.")
                return pd.DataFrame()
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Formato de arquivo não suportado.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

# Carregar dados
df = load_data(uploaded_file)

# Verifica se os dados foram carregados
if df.empty:
    st.warning("⚠️ Nenhum dado carregado. Por favor, envie um arquivo para análise.")
    st.stop()

# Renomear colunas para nomes padronizados (em inglês)
col_renames = {
    "Região": "Region",
    "Tipo_de_Solo": "Soil_Type",
    "Cultura": "Crop",
    "Condição_Climática": "Weather_Condition",
    "Fertilizante_Utilizado": "Fertilizer_Used",
    "Irrigação_Utilizada": "Irrigation_Used",
    "Produtividade_ton_ha": "Yield_tons_per_hectare",
    "Chuva_mm": "Rainfall_mm",
    "Temperatura_Celsius": "Temperature_Celsius"
}
df.rename(columns=col_renames, inplace=True)

# Verifica se as colunas essenciais existem
required_columns = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"Colunas ausentes no arquivo: {', '.join(missing_columns)}")
    st.stop()

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .filter-header {
        color: #1976D2;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Título do Dashboard
st.markdown('<h1 class="main-header">🌾 Dashboard de Análise Agrícola</h1>', unsafe_allow_html=True)

# Sidebar para filtros
st.sidebar.markdown('<div class="filter-header">🔍 Filtros</div>', unsafe_allow_html=True)

# Filtros
selected_region = st.sidebar.selectbox("Região", ["Todas as Regiões"] + sorted(df["Region"].dropna().unique().tolist()))
selected_soil_type = st.sidebar.selectbox("Tipo de Solo", ["Todos os Tipos"] + sorted(df["Soil_Type"].dropna().unique().tolist()))
selected_crop = st.sidebar.selectbox("Cultura", ["Todas as Culturas"] + sorted(df["Crop"].dropna().unique().tolist()))
selected_weather_condition = st.sidebar.selectbox("Condição Climática", ["Todas as Condições"] + sorted(df["Weather_Condition"].dropna().unique().tolist()))
selected_fertilizer_used = st.sidebar.selectbox("Uso de Fertilizante", ["Todos", "Sim", "Não"])
selected_irrigation_used = st.sidebar.selectbox("Uso de Irrigação", ["Todos", "Sim", "Não"])

# Botão para limpar filtros
if st.sidebar.button("🗑️ Limpar Filtros"):
    st.rerun()

# Aplicar filtros
filtered_df = df.copy()

if selected_region != "Todas as Regiões":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_soil_type != "Todos os Tipos":
    filtered_df = filtered_df[filtered_df["Soil_Type"] == selected_soil_type]
if selected_crop != "Todas as Culturas":
    filtered_df = filtered_df[filtered_df["Crop"] == selected_crop]
if selected_weather_condition != "Todas as Condições":
    filtered_df = filtered_df[filtered_df["Weather_Condition"] == selected_weather_condition]
if selected_fertilizer_used == "Sim":
    filtered_df = filtered_df[filtered_df["Fertilizer_Used"] == True]
elif selected_fertilizer_used == "Não":
    filtered_df = filtered_df[filtered_df["Fertilizer_Used"] == False]
if selected_irrigation_used == "Sim":
    filtered_df = filtered_df[filtered_df["Irrigation_Used"] == True]
elif selected_irrigation_used == "Não":
    filtered_df = filtered_df[filtered_df["Irrigation_Used"] == False]

# Exibir número total de registros
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Total de Registros:** {len(filtered_df)}")
st.sidebar.markdown(f"**📈 Total Original:** {len(df)}")

# Estatísticas
st.header("📊 Estatísticas")

if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_yield = filtered_df["Yield_tons_per_hectare"].mean()
        st.metric(
            label="🌾 Produtividade Média", 
            value=f"{avg_yield:.2f} ton/ha",
            delta=f"{avg_yield - df['Yield_tons_per_hectare'].mean():.2f}"
        )

    with col2:
        avg_rainfall = filtered_df["Rainfall_mm"].mean()
        st.metric(
            label="🌧️ Chuva Média", 
            value=f"{avg_rainfall:.2f} mm",
            delta=f"{avg_rainfall - df['Rainfall_mm'].mean():.2f}"
        )

    with col3:
        avg_temp = filtered_df["Temperature_Celsius"].mean()
        st.metric(
            label="🌡️ Temperatura Média", 
            value=f"{avg_temp:.2f} °C",
            delta=f"{avg_temp - df['Temperature_Celsius'].mean():.2f}"
        )

    with col4:
        corr_rainfall_yield = filtered_df["Rainfall_mm"].corr(filtered_df["Yield_tons_per_hectare"])
        st.metric(
            label="🔗 Correlação Chuva-Produtividade", 
            value=f"{corr_rainfall_yield:.3f}" if not pd.isna(corr_rainfall_yield) else "N/A"
        )
else:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")

# Visualizações
st.header("📈 Visualizações")

if not filtered_df.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Produtividade por Cultura")
        yield_by_crop = filtered_df.groupby("Crop")["Yield_tons_per_hectare"].mean().reset_index()
        yield_by_crop = yield_by_crop.sort_values("Yield_tons_per_hectare", ascending=False)

        fig_crop = px.bar(
            yield_by_crop, 
            x="Crop", 
            y="Yield_tons_per_hectare",
            title="Produtividade Média por Cultura",
            labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "Crop": "Cultura"},
            color="Yield_tons_per_hectare",
            color_continuous_scale="Greens"
        )
        fig_crop.update_layout(showlegend=False)
        st.plotly_chart(fig_crop, use_container_width=True)

    with col2:
        st.subheader("🗺️ Produtividade por Região")
        yield_by_region = filtered_df.groupby("Region")["Yield_tons_per_hectare"].mean().reset_index()
        yield_by_region = yield_by_region.sort_values("Yield_tons_per_hectare", ascending=False)

        fig_region = px.bar(
            yield_by_region, 
            x="Region", 
            y="Yield_tons_per_hectare",
            title="Produtividade Média por Região",
            labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "Region": "Região"},
            color="Yield_tons_per_hectare",
            color_continuous_scale="Blues"
        )
        fig_region.update_layout(showlegend=False)
        st.plotly_chart(fig_region, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏔️ Produtividade por Tipo de Solo")
        yield_by_soil = filtered_df.groupby("Soil_Type")["Yield_tons_per_hectare"].mean().reset_index()
        yield_by_soil = yield_by_soil.sort_values("Yield_tons_per_hectare", ascending=False)

        fig_soil = px.bar(
            yield_by_soil, 
            x="Soil_Type", 
            y="Yield_tons_per_hectare",
            title="Produtividade Média por Tipo de Solo",
            labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "Soil_Type": "Tipo de Solo"},
            color="Yield_tons_per_hectare",
            color_continuous_scale="Oranges"
        )
        fig_soil.update_layout(showlegend=False)
        fig_soil.update_xaxes(tickangle=45)
        st.plotly_chart(fig_soil, use_container_width=True)

    with col4:
        st.subheader("🌧️ Chuva vs Produtividade")
        fig_scatter = px.scatter(
            filtered_df, 
            x="Rainfall_mm", 
            y="Yield_tons_per_hectare",
            title="Correlação: Chuva vs Produtividade",
            labels={"Rainfall_mm": "Chuva (mm)", "Yield_tons_per_hectare": "Produtividade (ton/ha)"},
            color="Temperature_Celsius",
            color_continuous_scale="Viridis",
            hover_data=["Region", "Crop", "Soil_Type"]
        )

        z = np.polyfit(filtered_df["Rainfall_mm"], filtered_df["Yield_tons_per_hectare"], 1)
        p = np.poly1d(z)
        fig_scatter.add_traces(go.Scatter(
            x=filtered_df["Rainfall_mm"], 
            y=p(filtered_df["Rainfall_mm"]),
            mode="lines",
            name="Linha de Tendência",
            line=dict(color="red", dash="dash")
        ))

        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📊 Distribuição de Produtividade")
    fig_hist = px.histogram(
        filtered_df, 
        x="Yield_tons_per_hectare",
        nbins=30,
        title="Distribuição da Produtividade",
        labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "count": "Frequência"},
        color_discrete_sequence=["#4CAF50"]
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("📋 Ver Dados Filtrados"):
        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Baixar dados filtrados (CSV)",
            data=csv,
            file_name="dados_agricolas_filtrados.csv",
            mime="text/csv"
        )

else:
    st.info("ℹ️ Ajuste os filtros para ver as visualizações.")
# Resultado da melhor colheita com base nos filtros
st.header("🏆 Melhor Resultado de Produtividade")

if not filtered_df.empty:
    best_row = filtered_df.loc[filtered_df["Yield_tons_per_hectare"].idxmax()]

    st.markdown(f"""
    <style>
    .best-result {{
        background-color: var(--background-color-secondary);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid var(--primary-color);
        margin-top: 1rem;
        font-size: 1.05rem;
    }}
    .best-result h3 {{
        color: var(--primary-color);
    }}
   </style>

   <div class="best-result">
    <h3>🌟 A melhor colheita registrada com os filtros atuais foi:</h3>
    <ul>
        <li><strong>Região:</strong> {best_row['Region']}</li>
        <li><strong>Cultura:</strong> {best_row['Crop']}</li>
        <li><strong>Tipo de Solo:</strong> {best_row['Soil_Type']}</li>
        <li><strong>Condição Climática:</strong> {best_row['Weather_Condition']}</li>
        <li><strong>Uso de Fertilizante:</strong> {"Sim" if best_row["Fertilizer_Used"] else "Não"}</li>
        <li><strong>Uso de Irrigação:</strong> {"Sim" if best_row["Irrigation_Used"] else "Não"}</li>
        <li><strong>Chuva:</strong> {best_row["Rainfall_mm"]:.2f} mm</li>
        <li><strong>Temperatura:</strong> {best_row["Temperature_Celsius"]:.2f} °C</li>
        <li><strong>🌾 Produtividade:</strong> <strong>{best_row["Yield_tons_per_hectare"]:.2f} ton/ha</strong></li>
    </ul>
   </div>
  """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Nenhuma colheita disponível com os filtros atuais.")
    
# Ranking das 3 melhores colheitas com base nos filtros
st.header("🥇 Top 3 Colheitas com Maior Produtividade")

if not filtered_df.empty:
    top_3 = filtered_df.sort_values("Yield_tons_per_hectare", ascending=False).head(3).copy()
    top_3["Fertilizer_Used"] = top_3["Fertilizer_Used"].map({True: "Sim", False: "Não"})
    top_3["Irrigation_Used"] = top_3["Irrigation_Used"].map({True: "Sim", False: "Não"})

    top_3_display = top_3[[
        "Region", "Crop", "Soil_Type", "Weather_Condition",
        "Fertilizer_Used", "Irrigation_Used",
        "Rainfall_mm", "Temperature_Celsius", "Yield_tons_per_hectare"
    ]].rename(columns={
        "Region": "Região",
        "Crop": "Cultura",
        "Soil_Type": "Tipo de Solo",
        "Weather_Condition": "Condição Climática",
        "Fertilizer_Used": "Fertilizante",
        "Irrigation_Used": "Irrigação",
        "Rainfall_mm": "Chuva (mm)",
        "Temperature_Celsius": "Temperatura (°C)",
        "Yield_tons_per_hectare": "Produtividade (ton/ha)"
    })

    st.table(top_3_display.style.format({
        "Chuva (mm)": "{:.2f}",
        "Temperatura (°C)": "{:.2f}",
        "Produtividade (ton/ha)": "{:.2f}"
    }))
else:
    st.info("ℹ️ Nenhuma colheita disponível para exibir ranking com os filtros atuais.")
    
st.markdown("---")
# ----------------------------
# 🤖 Machine Learning: KNN + Métricas
# ----------------------------

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

st.header("🤖 Previsão de Produtividade com KNN")

# Verifica se as colunas necessárias estão presentes
knn_required = ["Rainfall_mm", "Temperature_Celsius", "Soil_Type", "Crop", "Yield_tons_per_hectare"]
missing_knn_cols = [col for col in knn_required if col not in filtered_df.columns]

if missing_knn_cols:
    st.warning(f"Colunas faltando para análise de Machine Learning: {', '.join(missing_knn_cols)}")
else:
    df_ml = filtered_df[knn_required].dropna().copy()

    if df_ml.empty:
        st.warning("⚠️ Dados insuficientes para treinamento do modelo.")
    else:
        # Encoding de variáveis categóricas
        label_encoder_soil = LabelEncoder()
        label_encoder_crop = LabelEncoder()
        df_ml["Soil_Type"] = label_encoder_soil.fit_transform(df_ml["Soil_Type"])
        df_ml["Crop"] = label_encoder_crop.fit_transform(df_ml["Crop"])

        # Variáveis independentes e alvo
        X = df_ml.drop("Yield_tons_per_hectare", axis=1)
        y = df_ml["Yield_tons_per_hectare"]

        # Treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinamento do modelo
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # Cálculo das métricas
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Exibição no dashboard
        col1, col2, col3 = st.columns(3)
        col1.metric("📈 R²", f"{r2:.2f}")
        col2.metric("📉 RMSE", f"{rmse:.2f} ton/ha")
        col3.metric("📏 MAE", f"{mae:.2f} ton/ha")

        # Comparação gráfica real vs previsto
        st.subheader("🔍 Comparação: Real vs Previsto (KNN)")
        comparison_df = pd.DataFrame({"Real": y_test, "Previsto": y_pred})
        fig_pred = px.scatter(
            comparison_df,
            x="Real",
            y="Previsto",
            title="Produtividade: Valores Reais vs. Previsto pelo KNN",
            labels={"Real": "Produtividade Real (ton/ha)", "Previsto": "Produtividade Prevista (ton/ha)"},
            color_discrete_sequence=["#2E7D32"]
        )
        fig_pred.add_trace(
            go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Ideal',
                line=dict(dash='dash', color='red')
            )
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
# Treinamento com Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Cálculo das métricas
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

st.subheader("🌲 Resultados com Random Forest")
col4, col5, col6 = st.columns(3)
col4.metric("📈 R²", f"{r2_rf:.2f}")
col5.metric("📉 RMSE", f"{rmse_rf:.2f} ton/ha")
col6.metric("📏 MAE", f"{mae_rf:.2f} ton/ha")

# Gráfico real vs previsto - Random Forest
comparison_rf_df = pd.DataFrame({"Real": y_test, "Previsto": y_pred_rf})
fig_rf = px.scatter(
    comparison_rf_df,
    x="Real",
    y="Previsto",
    title="Produtividade: Real vs Previsto (Random Forest)",
    labels={"Real": "Produtividade Real (ton/ha)", "Previsto": "Produtividade Prevista (ton/ha)"},
    color_discrete_sequence=["#1565C0"]
)
fig_rf.add_trace(
    go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Ideal',
        line=dict(dash='dash', color='red')
    )
)
st.plotly_chart(fig_rf, use_container_width=True)

# ----------------------------
# 🤖 Machine Learning: Comparação KNN vs RF
# ----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Bibliotecas de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Configurar matplotlib para suporte a fontes CJK
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Noto Sans CJK JP', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Esquema de cores especificado
CORES = ['#4080FF', '#57A9FB', '#37D4CF', '#23C343', '#FBE842', '#FF9A2E', '#A9AEB8']

def criar_dados_produtividade():
    """Cria dados simulados de produtividade agrícola"""
    np.random.seed(42)
    n_samples = 1000
    
    # Variáveis independentes (features)
    temperatura = np.random.normal(25, 5, n_samples)  # Temperatura média (°C)
    precipitacao = np.random.normal(800, 200, n_samples)  # Precipitação (mm)
    ph_solo = np.random.normal(6.5, 0.8, n_samples)  # pH do solo
    fertilizante = np.random.normal(150, 50, n_samples)  # Quantidade de fertilizante (kg/ha)
    
    # Variável dependente (target) - Produtividade
    # Fórmula simulada baseada nas variáveis independentes
    produtividade = (
        0.3 * temperatura + 
        0.002 * precipitacao + 
        2.0 * ph_solo + 
        0.01 * fertilizante + 
        np.random.normal(0, 2, n_samples)  # Ruído
    )
    
    # Garantir valores positivos e realistas
    produtividade = np.clip(produtividade, 1, 15)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'temperatura': temperatura,
        'precipitacao': precipitacao,
        'ph_solo': ph_solo,
        'fertilizante': fertilizante,
        'produtividade': produtividade
    })
    
    return df

def treinar_modelos(df):
    """Treina os modelos KNN e Random Forest"""
    # Preparar dados
    X = df[['temperatura', 'precipitacao', 'ph_solo', 'fertilizante']]
    y = df['produtividade']
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizar dados para KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar KNN
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    
    # Treinar Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # Calcular métricas
    metricas = {
        'KNN': {
            'r2': r2_score(y_test, y_pred_knn),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_knn)),
            'mae': mean_absolute_error(y_test, y_pred_knn),
            'y_pred': y_pred_knn
        },
        'Random Forest': {
            'r2': r2_score(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'y_pred': y_pred_rf
        }
    }
    
    return metricas, y_test

def configurar_estilo_plotly():
    """Configura o estilo visual para gráficos Plotly"""
    return {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#333333'},
        'xaxis': {
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': '#E0E0E0',
            'griddash': 'dash',
            'showline': True,
            'linewidth': 1,
            'linecolor': '#CCCCCC'
        },
        'yaxis': {
            'showgrid': True,
            'gridwidth': 1,
            'gridcolor': '#E0E0E0',
            'griddash': 'dash',
            'showline': True,
            'linewidth': 1,
            'linecolor': '#CCCCCC'
        }
    }

def criar_grafico_comparacao_geral(metricas):
    """Cria gráfico de barras agrupadas para comparação geral"""
    modelos = list(metricas.keys())
    r2_scores = [metricas[modelo]['r2'] for modelo in modelos]
    rmse_scores = [metricas[modelo]['rmse'] for modelo in modelos]
    mae_scores = [metricas[modelo]['mae'] for modelo in modelos]
    
    fig = go.Figure()
    
    # Adicionar barras para cada métrica
    fig.add_trace(go.Bar(
        name='R²',
        x=modelos,
        y=r2_scores,
        marker_color=CORES[0],
        text=[f'{score:.3f}' for score in r2_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>R²: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='RMSE',
        x=modelos,
        y=rmse_scores,
        marker_color=CORES[1],
        text=[f'{score:.2f}' for score in rmse_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f} ton/ha<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        name='MAE',
        x=modelos,
        y=mae_scores,
        marker_color=CORES[2],
        text=[f'{score:.2f}' for score in mae_scores],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>MAE: %{y:.2f} ton/ha<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Comparação de Desempenho: KNN vs Random Forest',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333333'}
        },
        xaxis_title='Modelos',
        yaxis_title='Valores das Métricas',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        **configurar_estilo_plotly()
    )
    
    return fig

def criar_graficos_individuais(metricas):
    """Cria gráficos individuais para cada métrica"""
    modelos = list(metricas.keys())
    
    # Gráfico R²
    fig_r2 = go.Figure(data=[
        go.Bar(
            x=modelos,
            y=[metricas[modelo]['r2'] for modelo in modelos],
            marker_color=[CORES[0], CORES[3]],
            text=[f'{metricas[modelo]["r2"]:.3f}' for modelo in modelos],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>R²: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig_r2.update_layout(
        title={
            'text': 'Coeficiente de Determinação (R²)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#333333'}
        },
        xaxis_title='Modelos',
        yaxis_title='R² Score',
        **configurar_estilo_plotly()
    )
    
    # Gráfico RMSE
    fig_rmse = go.Figure(data=[
        go.Bar(
            x=modelos,
            y=[metricas[modelo]['rmse'] for modelo in modelos],
            marker_color=[CORES[1], CORES[4]],
            text=[f'{metricas[modelo]["rmse"]:.2f}' for modelo in modelos],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>RMSE: %{y:.2f} ton/ha<extra></extra>'
        )
    ])
    
    fig_rmse.update_layout(
        title={
            'text': 'Erro Quadrático Médio (RMSE)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#333333'}
        },
        xaxis_title='Modelos',
        yaxis_title='RMSE (ton/ha)',
        **configurar_estilo_plotly()
    )
    
    # Gráfico MAE
    fig_mae = go.Figure(data=[
        go.Bar(
            x=modelos,
            y=[metricas[modelo]['mae'] for modelo in modelos],
            marker_color=[CORES[2], CORES[5]],
            text=[f'{metricas[modelo]["mae"]:.2f}' for modelo in modelos],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>MAE: %{y:.2f} ton/ha<extra></extra>'
        )
    ])
    
    fig_mae.update_layout(
        title={
            'text': 'Erro Absoluto Médio (MAE)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#333333'}
        },
        xaxis_title='Modelos',
        yaxis_title='MAE (ton/ha)',
        **configurar_estilo_plotly()
    )
    
    return fig_r2, fig_rmse, fig_mae

def criar_grafico_matplotlib(metricas):
    """Cria gráfico estático com matplotlib seguindo o código original"""
    modelos = list(metricas.keys())
    r2_scores = [metricas[modelo]['r2'] for modelo in modelos]
    rmse_scores = [metricas[modelo]['rmse'] for modelo in modelos]
    mae_scores = [metricas[modelo]['mae'] for modelo in modelos]
    
    x = np.arange(len(modelos))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Configurar estilo
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    ax.set_axisbelow(True)
    
    # Criar barras
    bars1 = ax.bar(x - width, r2_scores, width, label='R²', color=CORES[0], alpha=0.8)
    bars2 = ax.bar(x, rmse_scores, width, label='RMSE', color=CORES[1], alpha=0.8)
    bars3 = ax.bar(x + width, mae_scores, width, label='MAE', color=CORES[2], alpha=0.8)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('Valores das Métricas', fontsize=12)
    ax.set_title('Desempenho dos Modelos - KNN vs Random Forest', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/comparacao_modelos_matplotlib.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Função principal"""
    print("🚀 Iniciando análise de modelos de Machine Learning...")
    
    # Criar dados
    print("📊 Criando dados de produtividade agrícola...")
    df = criar_dados_produtividade()
    
    # Treinar modelos
    print("🤖 Treinando modelos KNN e Random Forest...")
    metricas, y_test = treinar_modelos(df)
    
    # Exibir resultados
    print("\n📈 Resultados dos Modelos:")
    for modelo, resultado in metricas.items():
        print(f"\n{modelo}:")
        print(f"  R²: {resultado['r2']:.3f}")
        print(f"  RMSE: {resultado['rmse']:.2f} ton/ha")
        print(f"  MAE: {resultado['mae']:.2f} ton/ha")
    
    # Criar gráficos
    print("\n🎨 Criando gráficos de comparação...")
    
    # Gráfico de comparação geral
    fig_comparacao = criar_grafico_comparacao_geral(metricas)
    fig_comparacao.write_html("/home/ubuntu/comparacao_geral.html")
    fig_comparacao.write_image("/home/ubuntu/comparacao_geral.png", width=1000, height=600)
    
    # Gráficos individuais
    fig_r2, fig_rmse, fig_mae = criar_graficos_individuais(metricas)
    
    fig_r2.write_html("/home/ubuntu/grafico_r2.html")
    fig_r2.write_image("/home/ubuntu/grafico_r2.png", width=800, height=500)
    
    fig_rmse.write_html("/home/ubuntu/grafico_rmse.html")
    fig_rmse.write_image("/home/ubuntu/grafico_rmse.png", width=800, height=500)
    
    fig_mae.write_html("/home/ubuntu/grafico_mae.html")
    fig_mae.write_image("/home/ubuntu/grafico_mae.png", width=800, height=500)
    
    # Gráfico matplotlib
    criar_grafico_matplotlib(metricas)
    
    print("\n✅ Todos os gráficos foram criados com sucesso!")
    print("\n📁 Arquivos gerados:")
    print("  • comparacao_geral.html (interativo)")
    print("  • comparacao_geral.png")
    print("  • grafico_r2.html (interativo)")
    print("  • grafico_r2.png")
    print("  • grafico_rmse.html (interativo)")
    print("  • grafico_rmse.png")
    print("  • grafico_mae.html (interativo)")
    print("  • grafico_mae.png")
    print("  • comparacao_modelos_matplotlib.png")

if __name__ == "__main__":
    main()

st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌾 Dashboard de Análise Agrícola | Desenvolvido por Sérgio</p>
    </div>
    """, 
    unsafe_allow_html=True
)

