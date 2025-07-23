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

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌾 Dashboard de Análise Agrícola | Desenvolvido por Sérgio</p>
    </div>
    """, 
    unsafe_allow_html=True
)
