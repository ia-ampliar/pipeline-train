# import plotly.graph_objects as go

# # Dados
# metodos = ['Resultado Bruto', 'Macenko']
# valores = [1287096, 703305]

# # Gráfico
# fig = go.Figure(data=[
#     go.Bar(
#         x=metodos,
#         y=valores,
#         text=valores,
#         textposition='auto',
#         textfont=dict(size=40)
#     )
# ])

# fig.update_layout(
#     title=dict(
#         text='Comparação de Número de Imagens Processadas',
#         x=0.5,
#         xanchor='center',
#         font=dict(size=24, family='Arial', color='black', weight='bold')
#     ),
#     xaxis=dict(
#         title=dict(
#             text='Método',
#             font=dict(size=20, family='Arial', color='black', weight='bold')
#         ),
#         tickfont=dict(size=18, family='Arial', color='black', weight='bold')
#     ),
#     yaxis=dict(
#         title=dict(
#             text='Número de Imagens',
#             font=dict(size=20, family='Arial', color='black', weight='bold')
#         ),
#         tickfont=dict(size=18, family='Arial', color='black', weight='bold')
#     ),
#     template='plotly_white'
# )

# fig.show()

import plotly.graph_objects as go

# Nomes dos conjuntos
conjuntos = ['Treinamento', 'Teste', 'Validação']

# Quantidades de CIN e noCIN
cin = [445624, 18128, 43027]
nocin = [87367, 20621, 75196]

# Gráfico de barras agrupadas
fig = go.Figure(data=[
    go.Bar(name='CIN', x=conjuntos, y=cin, text=cin, textposition='auto', textfont=dict(size=40)),
    go.Bar(name='noCIN', x=conjuntos, y=nocin, text=nocin, textposition='auto', textfont=dict(size=40))
])

# Layout
fig.update_layout(
    title=dict(
        text='Distribuição de Amostras por Conjunto e Classe',
        x=0.5,
        font=dict(size=24, family='Arial', color='black', weight='bold')
    ),
    xaxis=dict(
        title=dict(
            text='Conjunto',
            font=dict(size=20, family='Arial', color='black', weight='bold')
        ),
        tickfont=dict(size=18, family='Arial', color='black', weight='bold')
    ),
    yaxis=dict(
        title=dict(
            text='Número de Amostras',
            font=dict(size=20, family='Arial', color='black', weight='bold')
        ),
        tickfont=dict(size=18, family='Arial', color='black', weight='bold')
    ),
    barmode='group',
    template='plotly_white'
)

fig.show()
