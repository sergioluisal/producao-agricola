import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

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
        fig_crop.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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
        fig_region.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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
        fig_soil.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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

        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
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
    fig_hist.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
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
    <div style="background-color: #f0f2f6; color: #333; padding: 1.5rem; border-radius: 10px; border-left: 6px solid #4CAF50; margin-top: 1rem; font-size: 1.05rem;">
        <h3 style="color: #4CAF50;">🌟 A melhor colheita registrada com os filtros atuais foi:</h3>
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
# 🤖 SEÇÃO DE MACHINE LEARNING MELHORADA COM FUNDO TRANSPARENTE
# ----------------------------

# Funções auxiliares para gráficos interativos com fundo transparente
def get_plotly_style_transparent():
    """Configura o estilo visual para gráficos Plotly com fundo transparente"""
    return {
        'plot_bgcolor': 'rgba(0,0,0,0)',  # Fundo transparente
        'paper_bgcolor': 'rgba(0,0,0,0)',  # Fundo do papel transparente
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

def create_interactive_comparison_chart_transparent(metrics):
    """Cria gráfico de comparação interativo melhorado com fundo transparente"""
    # Esquema de cores especificado
    colors = ['#4080FF', '#57A9FB', '#37D4CF', '#23C343', '#FBE842', '#FF9A2E', '#A9AEB8']
    
    modelos = list(metrics.keys())
    r2_scores = [metrics[modelo]['r2'] for modelo in modelos]
    rmse_scores = [metrics[modelo]['rmse'] for modelo in modelos]
    mae_scores = [metrics[modelo]['mae'] for modelo in modelos]
    
    fig = go.Figure()
    
    # Adicionar barras para cada métrica
    fig.add_trace(go.Bar(
        name='R² (Coeficiente de Determinação)',
        x=modelos,
        y=r2_scores,
        marker_color=colors[0],
        text=[f'{score:.3f}' for score in r2_scores],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' +
                     'R²: %{y:.3f}<br>' +
                     '<i>Explica %{customdata:.1f}% da variância</i>' +
                     '<extra></extra>',
        customdata=[score * 100 for score in r2_scores],
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        name='RMSE (Erro Quadrático Médio)',
        x=modelos,
        y=rmse_scores,
        marker_color=colors[1],
        text=[f'{score:.2f}' for score in rmse_scores],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' +
                     'RMSE: %{y:.2f} ton/ha<br>' +
                     '<i>Erro médio quadrático</i>' +
                     '<extra></extra>',
        offsetgroup=2
    ))
    
    fig.add_trace(go.Bar(
        name='MAE (Erro Absoluto Médio)',
        x=modelos,
        y=mae_scores,
        marker_color=colors[2],
        text=[f'{score:.2f}' for score in mae_scores],
        textposition='auto',
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{x}</b><br>' +
                     'MAE: %{y:.2f} ton/ha<br>' +
                     '<i>Erro médio absoluto</i>' +
                     '<extra></extra>',
        offsetgroup=3
    ))
    
    # Adicionar anotações para melhor modelo
    annotations = []

    # Determinar melhor modelo para cada métrica
    melhor_r2_idx = np.argmax(r2_scores)
    melhor_rmse_idx = np.argmin(rmse_scores)

    # Anotação para melhor R²
    annotations.append(dict(
        x=modelos[melhor_r2_idx],
        y=r2_scores[melhor_r2_idx] + max(r2_scores) * 0.1,
        text="🏆 Melhor R²",
        showarrow=True,
        arrowhead=2,
        arrowcolor=colors[0],
        font=dict(color=colors[0], size=10)
    ))

    # Anotação para melhor RMSE
    annotations.append(dict(
        x=modelos[melhor_rmse_idx],
        y=rmse_scores[melhor_rmse_idx] + max(rmse_scores) * 0.1,
        text="🎯 Menor RMSE",
        showarrow=True,
        arrowhead=2,
        arrowcolor=colors[1],
        font=dict(color=colors[1], size=10)
    ))

    # Layout principal do gráfico
    fig.update_layout(
        title={
            'text': '🤖 Comparação Interativa: KNN vs Random Forest<br>' +
                    '<sub>Análise de Desempenho para Predição de Produtividade Agrícola</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#FFFFFF'}  # Ajuste conforme tema
        },
        xaxis_title='Modelos de Machine Learning',
        yaxis_title='Valores das Métricas',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        height=700,
        annotations=annotations,
        **get_plotly_style_transparent()
    )

    # Anotação explicando as métricas
    fig.add_annotation(
        text="📘 <b>R²</b>: Coef. Determinação &nbsp;&nbsp;&nbsp; 🔵 <b>RMSE</b>: Erro Quadrático Médio &nbsp;&nbsp;&nbsp; 🟢 <b>MAE</b>: Erro Absoluto Médio",
        xref="paper", yref="paper",
        x=0, y=0.98,
        showarrow=False,
        align="left",
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0)"
    )

    # Botões para alternar visibilidade das métricas
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True, True]}],
                        label="Todas as Métricas",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [True, False, False]}],
                        label="Apenas R²",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False, True, False]}],
                        label="Apenas RMSE",
                        method="restyle"
                    ),
                    dict(
                        args=[{"visible": [False, False, True]}],
                        label="Apenas MAE",
                        method="restyle"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=0.94,
                yanchor="top"
            ),
        ]
    )

    return fig

def create_dashboard_comparison_transparent(metrics, y_test):
    """Cria dashboard completo com múltiplos gráficos com fundo transparente"""
    colors = ['#4080FF', '#57A9FB', '#37D4CF', '#23C343']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Comparação de Métricas R²',
            'Distribuição de Erros',
            'Real vs Predito - KNN',
            'Real vs Predito - Random Forest'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Gráfico de barras de comparação R²
    modelos = list(metrics.keys())
    r2_scores = [metrics[modelo]['r2'] for modelo in modelos]
    
    fig.add_trace(
        go.Bar(
            name='R²', 
            x=modelos, 
            y=r2_scores, 
            marker_color=colors[0],
            text=[f'{score:.3f}' for score in r2_scores],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Distribuição de erros
    erros_knn = np.abs(y_test - metrics['KNN']['y_pred'])
    erros_rf = np.abs(y_test - metrics['Random Forest']['y_pred'])
    
    fig.add_trace(
        go.Histogram(x=erros_knn, name='Erros KNN', marker_color=colors[0], opacity=0.7, nbinsx=20),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=erros_rf, name='Erros RF', marker_color=colors[3], opacity=0.7, nbinsx=20),
        row=1, col=2
    )
    
    # 3. Scatter KNN
    fig.add_trace(
        go.Scatter(
            x=y_test, y=metrics['KNN']['y_pred'],
            mode='markers', name='KNN',
            marker=dict(color=colors[0], size=4, opacity=0.6),
            hovertemplate='<b>KNN</b><br>Real: %{x:.2f}<br>Predito: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Scatter Random Forest
    fig.add_trace(
        go.Scatter(
            x=y_test, y=metrics['Random Forest']['y_pred'],
            mode='markers', name='Random Forest',
            marker=dict(color=colors[3], size=4, opacity=0.6),
            hovertemplate='<b>Random Forest</b><br>Real: %{x:.2f}<br>Predito: %{y:.2f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Linhas ideais para os scatters
    min_val = min(y_test.min(), metrics['KNN']['y_pred'].min(), metrics['Random Forest']['y_pred'].min())
    max_val = max(y_test.max(), metrics['KNN']['y_pred'].max(), metrics['Random Forest']['y_pred'].max())
    
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Ideal',
                line=dict(dash='dash', color='red', width=2),
                showlegend=(col == 1),
                hovertemplate='Linha Ideal<extra></extra>'
            ),
            row=2, col=col
        )
    
    fig.update_layout(
        title={
            'text': '📊 Dashboard Completo: Análise KNN vs Random Forest',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#FFFFFF'}
        },
        height=800,
        **get_plotly_style_transparent()
    )
    
    # Atualizar eixos
    fig.update_xaxes(title_text="Modelos", row=1, col=1)
    fig.update_yaxes(title_text="R² Score", row=1, col=1)
    fig.update_xaxes(title_text="Erro Absoluto", row=1, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    fig.update_xaxes(title_text="Produtividade Real (ton/ha)", row=2, col=1)
    fig.update_yaxes(title_text="Produtividade Predita (ton/ha)", row=2, col=1)
    fig.update_xaxes(title_text="Produtividade Real (ton/ha)", row=2, col=2)
    fig.update_yaxes(title_text="Produtividade Predita (ton/ha)", row=2, col=2)
    
    return fig

# SEÇÃO PRINCIPAL DE MACHINE LEARNING
st.markdown('<div class="ml-section">', unsafe_allow_html=True)
st.markdown('<div class="comparison-header">🤖 Machine Learning: Comparação Interativa KNN vs Random Forest</div>', unsafe_allow_html=True)

# Sidebar para configurações ML
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configurações ML")
knn_neighbors = st.sidebar.slider("KNN - Número de Vizinhos", 3, 15, 5, 1)
rf_estimators = st.sidebar.slider("Random Forest - Número de Árvores", 50, 200, 100, 25)
test_size = st.sidebar.slider("Tamanho do Conjunto de Teste (%)", 10, 40, 20, 5) / 100

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
        # Imports necessários
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # Encoding de variáveis categóricas
        label_encoder_soil = LabelEncoder()
        label_encoder_crop = LabelEncoder()
        df_ml["Soil_Type"] = label_encoder_soil.fit_transform(df_ml["Soil_Type"])
        df_ml["Crop"] = label_encoder_crop.fit_transform(df_ml["Crop"])

        # Variáveis independentes e alvo
        X = df_ml.drop("Yield_tons_per_hectare", axis=1)
        y = df_ml["Yield_tons_per_hectare"]

        # Treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Normalizar dados para KNN
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Treinamento dos modelos
        with st.spinner("🔄 Treinando modelos KNN e Random Forest..."):
            # KNN
            knn = KNeighborsRegressor(n_neighbors=knn_neighbors)
            knn.fit(X_train_scaled, y_train)
            y_pred_knn = knn.predict(X_test_scaled)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=rf_estimators, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

        # Cálculo das métricas
        metrics = {
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

        # Exibição das métricas em cards melhorados
        st.subheader("📊 Resultados dos Modelos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔍 K-Nearest Neighbors (KNN)")
            knn_col1, knn_col2, knn_col3 = st.columns(3)
            knn_col1.metric("📈 R²", f"{metrics['KNN']['r2']:.3f}")
            knn_col2.metric("📉 RMSE", f"{metrics['KNN']['rmse']:.2f} ton/ha")
            knn_col3.metric("📏 MAE", f"{metrics['KNN']['mae']:.2f} ton/ha")
        
        with col2:
            st.markdown("#### 🌲 Random Forest")
            rf_col1, rf_col2, rf_col3 = st.columns(3)
            rf_col1.metric("📈 R²", f"{metrics['Random Forest']['r2']:.3f}")
            rf_col2.metric("📉 RMSE", f"{metrics['Random Forest']['rmse']:.2f} ton/ha")
            rf_col3.metric("📏 MAE", f"{metrics['Random Forest']['mae']:.2f} ton/ha")

        # Análise comparativa
        st.subheader("🏆 Análise Comparativa")
        melhor_r2 = max(metrics, key=lambda x: metrics[x]['r2'])
        melhor_rmse = min(metrics, key=lambda x: metrics[x]['rmse'])
        melhor_mae = min(metrics, key=lambda x: metrics[x]['mae'])
        
        col1, col2, col3 = st.columns(3)
        col1.success(f"🏆 Melhor R²: **{melhor_r2}** ({metrics[melhor_r2]['r2']:.3f})")
        col2.success(f"🎯 Menor RMSE: **{melhor_rmse}** ({metrics[melhor_rmse]['rmse']:.2f})")
        col3.success(f"📏 Menor MAE: **{melhor_mae}** ({metrics[melhor_mae]['mae']:.2f})")

        # GRÁFICO PRINCIPAL INTERATIVO MELHORADO COM FUNDO TRANSPARENTE
        st.subheader("🎨 Gráfico de Comparação Interativo")
        fig_comparison = create_interactive_comparison_chart_transparent(metrics)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # DASHBOARD COMPLETO COM FUNDO TRANSPARENTE
        st.subheader("📋 Dashboard Completo de Análise")
        fig_dashboard = create_dashboard_comparison_transparent(metrics, y_test)
        st.plotly_chart(fig_dashboard, use_container_width=True)

        # Gráficos individuais melhorados com fundo transparente
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 KNN: Real vs Predito")
            comparison_knn_df = pd.DataFrame({"Real": y_test, "Previsto": y_pred_knn})
            fig_knn = px.scatter(
                comparison_knn_df,
                x="Real",
                y="Previsto",
                title="KNN: Produtividade Real vs Predita",
                labels={"Real": "Produtividade Real (ton/ha)", "Previsto": "Produtividade Predita (ton/ha)"},
                color_discrete_sequence=["#4080FF"]
            )
            fig_knn.add_trace(
                go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Ideal',
                    line=dict(dash='dash', color='red')
                )
            )
            fig_knn.update_layout(**get_plotly_style_transparent())
            st.plotly_chart(fig_knn, use_container_width=True)

        with col2:
            st.subheader("🌲 Random Forest: Real vs Predito")
            comparison_rf_df = pd.DataFrame({"Real": y_test, "Previsto": y_pred_rf})
            fig_rf = px.scatter(
                comparison_rf_df,
                x="Real",
                y="Previsto",
                title="Random Forest: Produtividade Real vs Predita",
                labels={"Real": "Produtividade Real (ton/ha)", "Previsto": "Produtividade Predita (ton/ha)"},
                color_discrete_sequence=["#23C343"]
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
            fig_rf.update_layout(**get_plotly_style_transparent())
            st.plotly_chart(fig_rf, use_container_width=True)

        # Interpretação dos resultados
        with st.expander("💡 Interpretação dos Resultados"):
            st.markdown("""
            **Explicação das Métricas:**
            - **R² (Coeficiente de Determinação)**: Indica a proporção da variância explicada pelo modelo (0-1, quanto maior melhor)
            - **RMSE (Erro Quadrático Médio)**: Penaliza mais os erros grandes (quanto menor melhor)
            - **MAE (Erro Absoluto Médio)**: Média dos erros absolutos (quanto menor melhor)
            
            **Como Interpretar os Gráficos:**
            - **Gráfico de Comparação**: Permite filtrar métricas específicas usando os botões
            - **Real vs Predito**: Pontos próximos à linha vermelha indicam predições mais precisas
            - **Dashboard Completo**: Visão abrangente com múltiplas perspectivas dos resultados
            """)

        # Download dos resultados
        with st.expander("💾 Download dos Resultados"):
            # Criar DataFrame com resultados
            results_df = pd.DataFrame({
                'Modelo': list(metrics.keys()),
                'R²': [metrics[modelo]['r2'] for modelo in metrics.keys()],
                'RMSE': [metrics[modelo]['rmse'] for modelo in metrics.keys()],
                'MAE': [metrics[modelo]['mae'] for modelo in metrics.keys()]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Download CSV
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar Métricas (CSV)",
                data=csv_results,
                file_name="metricas_ml_comparacao.csv",
                mime="text/csv"
            )
            
            # Download predições
            predictions_df = pd.DataFrame({
                'Real': y_test,
                'KNN_Predito': y_pred_knn,
                'RF_Predito': y_pred_rf
            })
            csv_predictions = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Baixar Predições (CSV)",
                data=csv_predictions,
                file_name="predicoes_ml_comparacao.csv",
                mime="text/csv"
            )

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🌾 Dashboard de Análise Agrícola | Desenvolvido por Sérgio</p>
    </div>
    """, 
    unsafe_allow_html=True
)
