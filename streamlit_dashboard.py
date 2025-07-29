import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import traceback
import sys

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Análise Agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função de debug
def debug_info(message, data=None):
    """Função para debug - pode ser desabilitada em produção"""
    if st.sidebar.checkbox("🐛 Modo Debug", value=False):
        st.sidebar.write(f"DEBUG: {message}")
        if data is not None:
            st.sidebar.write(data)

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
    .ml-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 6px solid #4080FF;
        margin: 2rem 0;
    }
    .comparison-header {
        color: #4080FF;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .error-container {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título do Dashboard
st.markdown('<h1 class="main-header">🌾 Dashboard de Análise Agrícola</h1>', unsafe_allow_html=True)

# Função para carregar dados com tratamento robusto de erros
@st.cache_data
def load_data_robust(uploaded_file):
    """Carrega dados com tratamento robusto de erros"""
    if uploaded_file is None:
        return pd.DataFrame(), "Nenhum arquivo carregado"

    try:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        debug_info(f"Carregando arquivo: {uploaded_file.name}, extensão: {file_extension}")
        
        if file_extension == "csv":
            encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
            df = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                    encoding_used = encoding
                    debug_info(f"Arquivo carregado com encoding: {encoding}")
                    break
                except Exception as e:
                    debug_info(f"Falha com encoding {encoding}: {str(e)}")
                    continue
            
            if df is None:
                return pd.DataFrame(), "Não foi possível decodificar o arquivo CSV com nenhum encoding testado"
                
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)
            debug_info("Arquivo Excel carregado com sucesso")
        else:
            return pd.DataFrame(), f"Formato de arquivo não suportado: {file_extension}"
        
        # Verificações básicas do DataFrame
        if df.empty:
            return pd.DataFrame(), "Arquivo carregado está vazio"
        
        debug_info(f"DataFrame carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
        debug_info("Colunas encontradas:", list(df.columns))
        
        return df, "Sucesso"
        
    except Exception as e:
        error_msg = f"Erro ao carregar dados: {str(e)}"
        debug_info(f"ERRO: {error_msg}")
        debug_info("Traceback:", traceback.format_exc())
        return pd.DataFrame(), error_msg

# Upload de arquivo
st.subheader("📂 Carregamento de Dados")
uploaded_file = st.file_uploader("Carregue seu arquivo de dados (CSV, XLSX)", type=["csv", "xls", "xlsx"])

# Carregar dados
df, load_status = load_data_robust(uploaded_file)

if load_status != "Sucesso":
    st.error(f"❌ {load_status}")
    if df.empty:
        st.info("ℹ️ Por favor, carregue um arquivo válido para continuar.")
        st.stop()

# Verificar se dados foram carregados
if df.empty:
    st.warning("⚠️ Nenhum dado carregado. Por favor, envie um arquivo para análise.")
    st.stop()

# Mostrar informações básicas do dataset
with st.expander("ℹ️ Informações do Dataset"):
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Linhas", df.shape[0])
    col2.metric("📋 Colunas", df.shape[1])
    col3.metric("💾 Tamanho", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    st.write("**Colunas disponíveis:**")
    st.write(list(df.columns))
    
    st.write("**Primeiras 5 linhas:**")
    st.dataframe(df.head())

# Renomear colunas para nomes padronizados
try:
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
    
    # Renomear apenas colunas que existem
    existing_renames = {k: v for k, v in col_renames.items() if k in df.columns}
    df.rename(columns=existing_renames, inplace=True)
    debug_info(f"Colunas renomeadas: {existing_renames}")
    
except Exception as e:
    st.error(f"Erro ao renomear colunas: {str(e)}")

# Verificar colunas essenciais
required_columns = ["Region", "Soil_Type", "Crop", "Weather_Condition"]
available_columns = [col for col in required_columns if col in df.columns]
missing_columns = [col for col in required_columns if col not in df.columns]

debug_info(f"Colunas disponíveis: {available_columns}")
debug_info(f"Colunas faltando: {missing_columns}")

if missing_columns:
    st.warning(f"⚠️ Algumas colunas essenciais estão ausentes: {', '.join(missing_columns)}")
    st.info("💡 O dashboard funcionará com as colunas disponíveis, mas algumas funcionalidades podem ser limitadas.")

# Sidebar para filtros
st.sidebar.markdown('<div class="filter-header">🔍 Filtros</div>', unsafe_allow_html=True)

# Aplicar filtros apenas para colunas que existem
filtered_df = df.copy()

try:
    if "Region" in df.columns:
        regions = ["Todas as Regiões"] + sorted(df["Region"].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Região", regions)
        if selected_region != "Todas as Regiões":
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    if "Soil_Type" in df.columns:
        soil_types = ["Todos os Tipos"] + sorted(df["Soil_Type"].dropna().unique().tolist())
        selected_soil_type = st.sidebar.selectbox("Tipo de Solo", soil_types)
        if selected_soil_type != "Todos os Tipos":
            filtered_df = filtered_df[filtered_df["Soil_Type"] == selected_soil_type]

    if "Crop" in df.columns:
        crops = ["Todas as Culturas"] + sorted(df["Crop"].dropna().unique().tolist())
        selected_crop = st.sidebar.selectbox("Cultura", crops)
        if selected_crop != "Todas as Culturas":
            filtered_df = filtered_df[filtered_df["Crop"] == selected_crop]

    if "Weather_Condition" in df.columns:
        weather_conditions = ["Todas as Condições"] + sorted(df["Weather_Condition"].dropna().unique().tolist())
        selected_weather_condition = st.sidebar.selectbox("Condição Climática", weather_conditions)
        if selected_weather_condition != "Todas as Condições":
            filtered_df = filtered_df[filtered_df["Weather_Condition"] == selected_weather_condition]

    if "Fertilizer_Used" in df.columns:
        selected_fertilizer_used = st.sidebar.selectbox("Uso de Fertilizante", ["Todos", "Sim", "Não"])
        if selected_fertilizer_used == "Sim":
            filtered_df = filtered_df[filtered_df["Fertilizer_Used"] == True]
        elif selected_fertilizer_used == "Não":
            filtered_df = filtered_df[filtered_df["Fertilizer_Used"] == False]

    if "Irrigation_Used" in df.columns:
        selected_irrigation_used = st.sidebar.selectbox("Uso de Irrigação", ["Todos", "Sim", "Não"])
        if selected_irrigation_used == "Sim":
            filtered_df = filtered_df[filtered_df["Irrigation_Used"] == True]
        elif selected_irrigation_used == "Não":
            filtered_df = filtered_df[filtered_df["Irrigation_Used"] == False]

except Exception as e:
    st.error(f"Erro ao aplicar filtros: {str(e)}")
    filtered_df = df.copy()

# Botão para limpar filtros
if st.sidebar.button("🗑️ Limpar Filtros"):
    st.rerun()

# Exibir número total de registros
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Total de Registros:** {len(filtered_df)}")
st.sidebar.markdown(f"**📈 Total Original:** {len(df)}")

# Estatísticas
st.header("📊 Estatísticas")

if not filtered_df.empty:
    try:
        col1, col2, col3, col4 = st.columns(4)

        # Verificar se as colunas necessárias existem antes de calcular métricas
        if "Yield_tons_per_hectare" in filtered_df.columns:
            with col1:
                avg_yield = filtered_df["Yield_tons_per_hectare"].mean()
                delta_yield = avg_yield - df["Yield_tons_per_hectare"].mean() if "Yield_tons_per_hectare" in df.columns else 0
                st.metric(
                    label="🌾 Produtividade Média", 
                    value=f"{avg_yield:.2f} ton/ha",
                    delta=f"{delta_yield:.2f}"
                )

        if "Rainfall_mm" in filtered_df.columns:
            with col2:
                avg_rainfall = filtered_df["Rainfall_mm"].mean()
                delta_rainfall = avg_rainfall - df["Rainfall_mm"].mean() if "Rainfall_mm" in df.columns else 0
                st.metric(
                    label="🌧️ Chuva Média", 
                    value=f"{avg_rainfall:.2f} mm",
                    delta=f"{delta_rainfall:.2f}"
                )

        if "Temperature_Celsius" in filtered_df.columns:
            with col3:
                avg_temp = filtered_df["Temperature_Celsius"].mean()
                delta_temp = avg_temp - df["Temperature_Celsius"].mean() if "Temperature_Celsius" in df.columns else 0
                st.metric(
                    label="🌡️ Temperatura Média", 
                    value=f"{avg_temp:.2f} °C",
                    delta=f"{delta_temp:.2f}"
                )

        if "Rainfall_mm" in filtered_df.columns and "Yield_tons_per_hectare" in filtered_df.columns:
            with col4:
                corr_rainfall_yield = filtered_df["Rainfall_mm"].corr(filtered_df["Yield_tons_per_hectare"])
                st.metric(
                    label="🔗 Correlação Chuva-Produtividade", 
                    value=f"{corr_rainfall_yield:.3f}" if not pd.isna(corr_rainfall_yield) else "N/A"
                )

    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {str(e)}")
        debug_info("Erro nas estatísticas:", traceback.format_exc())
else:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")

# Visualizações
st.header("📈 Visualizações")

if not filtered_df.empty:
    try:
        # Função para criar gráficos com tratamento de erro
        def create_safe_chart(chart_func, title, error_msg):
            try:
                return chart_func()
            except Exception as e:
                st.error(f"Erro ao criar {title}: {str(e)}")
                debug_info(f"Erro no gráfico {title}:", traceback.format_exc())
                return None

        col1, col2 = st.columns(2)

        # Gráfico 1: Produtividade por Cultura
        if "Crop" in filtered_df.columns and "Yield_tons_per_hectare" in filtered_df.columns:
            with col1:
                st.subheader("🌱 Produtividade por Cultura")
                
                def create_crop_chart():
                    yield_by_crop = filtered_df.groupby("Crop")["Yield_tons_per_hectare"].mean().reset_index()
                    yield_by_crop = yield_by_crop.sort_values("Yield_tons_per_hectare", ascending=False)
                    
                    fig = px.bar(
                        yield_by_crop, 
                        x="Crop", 
                        y="Yield_tons_per_hectare",
                        title="Produtividade Média por Cultura",
                        labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "Crop": "Cultura"},
                        color="Yield_tons_per_hectare",
                        color_continuous_scale="Greens"
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    return fig
                
                fig = create_safe_chart(create_crop_chart, "Produtividade por Cultura", "gráfico de culturas")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Gráfico 2: Produtividade por Região
        if "Region" in filtered_df.columns and "Yield_tons_per_hectare" in filtered_df.columns:
            with col2:
                st.subheader("🗺️ Produtividade por Região")
                
                def create_region_chart():
                    yield_by_region = filtered_df.groupby("Region")["Yield_tons_per_hectare"].mean().reset_index()
                    yield_by_region = yield_by_region.sort_values("Yield_tons_per_hectare", ascending=False)
                    
                    fig = px.bar(
                        yield_by_region, 
                        x="Region", 
                        y="Yield_tons_per_hectare",
                        title="Produtividade Média por Região",
                        labels={"Yield_tons_per_hectare": "Produtividade (ton/ha)", "Region": "Região"},
                        color="Yield_tons_per_hectare",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    return fig
                
                fig = create_safe_chart(create_region_chart, "Produtividade por Região", "gráfico de regiões")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Gráfico 3: Scatter plot se ambas as colunas existirem
        if "Rainfall_mm" in filtered_df.columns and "Yield_tons_per_hectare" in filtered_df.columns:
            st.subheader("🌧️ Chuva vs Produtividade")
            
            def create_scatter_chart():
                fig = px.scatter(
                    filtered_df, 
                    x="Rainfall_mm", 
                    y="Yield_tons_per_hectare",
                    title="Correlação: Chuva vs Produtividade",
                    labels={"Rainfall_mm": "Chuva (mm)", "Yield_tons_per_hectare": "Produtividade (ton/ha)"},
                    color="Temperature_Celsius" if "Temperature_Celsius" in filtered_df.columns else None,
                    color_continuous_scale="Viridis",
                    hover_data=["Region", "Crop", "Soil_Type"] if all(col in filtered_df.columns for col in ["Region", "Crop", "Soil_Type"]) else None
                )
                
                # Adicionar linha de tendência se possível
                try:
                    z = np.polyfit(filtered_df["Rainfall_mm"], filtered_df["Yield_tons_per_hectare"], 1)
                    p = np.poly1d(z)
                    fig.add_traces(go.Scatter(
                        x=filtered_df["Rainfall_mm"], 
                        y=p(filtered_df["Rainfall_mm"]),
                        mode="lines",
                        name="Linha de Tendência",
                        line=dict(color="red", dash="dash")
                    ))
                except:
                    pass  # Se não conseguir calcular a linha de tendência, continua sem ela
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                return fig
            
            fig = create_safe_chart(create_scatter_chart, "Scatter Chuva vs Produtividade", "gráfico de correlação")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erro geral nas visualizações: {str(e)}")
        debug_info("Erro geral:", traceback.format_exc())

# Seção de dados filtrados
with st.expander("📋 Ver Dados Filtrados"):
    try:
        st.dataframe(filtered_df, use_container_width=True)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Baixar dados filtrados (CSV)",
            data=csv,
            file_name="dados_agricolas_filtrados.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Erro ao exibir dados filtrados: {str(e)}")

# Seção de Machine Learning (apenas se as colunas necessárias existirem)
ml_required = ["Rainfall_mm", "Temperature_Celsius", "Soil_Type", "Crop", "Yield_tons_per_hectare"]
ml_available = [col for col in ml_required if col in filtered_df.columns]

if len(ml_available) >= 3:  # Pelo menos 3 colunas necessárias
    st.markdown("---")
    st.markdown('<div class="ml-section">', unsafe_allow_html=True)
    st.markdown('<div class="comparison-header">🤖 Machine Learning: Análise Básica</div>', unsafe_allow_html=True)
    
    try:
        # Sidebar para configurações ML
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚙️ Configurações ML")
        
        # Verificar se temos dados suficientes
        df_ml = filtered_df[ml_available].dropna().copy()
        
        if len(df_ml) < 10:
            st.warning("⚠️ Dados insuficientes para análise de Machine Learning (mínimo 10 registros).")
        else:
            st.success(f"✅ Dados disponíveis para ML: {len(df_ml)} registros com {len(ml_available)} variáveis")
            
            # Mostrar quais colunas estão sendo usadas
            st.info(f"📊 Variáveis disponíveis: {', '.join(ml_available)}")
            
            if "Yield_tons_per_hectare" in ml_available:
                # Estatísticas básicas da variável alvo
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Produtividade Média", f"{df_ml['Yield_tons_per_hectare'].mean():.2f}")
                col2.metric("📊 Desvio Padrão", f"{df_ml['Yield_tons_per_hectare'].std():.2f}")
                col3.metric("📏 Amplitude", f"{df_ml['Yield_tons_per_hectare'].max() - df_ml['Yield_tons_per_hectare'].min():.2f}")
    
    except Exception as e:
        st.error(f"Erro na seção de Machine Learning: {str(e)}")
        debug_info("Erro ML:", traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info(f"ℹ️ Seção de Machine Learning não disponível. Colunas necessárias: {', '.join(ml_required)}. Disponíveis: {', '.join(ml_available)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌾 Dashboard de Análise Agrícola | Versão Robusta com Tratamento de Erros</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Informações de debug no final (se habilitado)
if st.sidebar.checkbox("🐛 Modo Debug", value=False):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Informações de Debug")
    st.sidebar.write(f"**Python:** {sys.version}")
    st.sidebar.write(f"**Streamlit:** {st.__version__}")
    st.sidebar.write(f"**Pandas:** {pd.__version__}")
    st.sidebar.write(f"**Plotly:** {px.__version__}")
    st.sidebar.write(f"**NumPy:** {np.__version__}")
