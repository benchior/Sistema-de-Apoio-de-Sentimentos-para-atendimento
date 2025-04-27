import logging
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
from sentiment_analysis.analyzer import SentimentLabel

logger = logging.getLogger(__name__)

class SentimentDashboard:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.app = Dash(__name__)
        self.data = pd.DataFrame()
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Análise de Sentimentos - Atendimento ao Cliente", className="header"),

            dcc.Tabs([
                dcc.Tab(label="Análise em Tempo Real", children=[
                    html.Div([
                        dcc.Textarea(
                            id='input-text',
                            placeholder='Digite o feedback do cliente...',
                            style={'width': '100%', 'height': 100}
                        ),
                        html.Button('Analisar', id='analyze-button'),
                        html.Div(id='output-result')
                    ])
                ]),

                dcc.Tab(label="Visualizações", children=[
                    dcc.Graph(id='sentiment-distribution'),
                    dcc.Graph(id='priority-distribution'),
                    dcc.Graph(id='time-trend')
                ]),

                dcc.Tab(label="Dados Detalhados", children=[
                    dash_table.DataTable(
                        id='data-table',
                        columns=[
                            {'name': 'Texto', 'id': 'texto'},
                            {'name': 'Sentimento', 'id': 'sentimento'},
                            {'name': 'Prioridade', 'id': 'prioridade'},
                            {'name': 'Confiança', 'id': 'confianca'},
                            {'name': 'Palavras-chave', 'id': 'palavras_chave'},
                            {'name': 'Timestamp', 'id': 'timestamp'}
                        ],
                        page_size=10
                    )
                ])
            ]),

            dcc.Store(id='analysis-data')
        ])

    def _setup_callbacks(self):
        @self.app.callback(
            Output('output-result', 'children'),
            Output('analysis-data', 'data'),
            Input('analyze-button', 'n_clicks'),
            State('input-text', 'value')
        )
        def analyze_text(n_clicks, texto):
            if n_clicks and texto:
                try:
                    result = self.analyzer.analyze_text(texto)
                    new_row = {
                        'texto': result.texto,
                        'sentimento': result.sentimento.name,
                        'prioridade': result.prioridade.name,
                        'confianca': result.confianca,
                        'palavras_chave': ', '.join(result.palavras_chave),
                        'timestamp': result.timestamp.isoformat()
                    }

                    self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)

                    alert_color = {
                        SentimentLabel.POSITIVO: 'green',
                        SentimentLabel.NEUTRO: 'blue',
                        SentimentLabel.NEGATIVO: 'orange',
                        SentimentLabel.CRITICO: 'red'
                    }.get(result.sentimento, 'blue')

                    return [
                        html.Div([
                            html.H3("Resultado da Análise:"),
                            html.P(f"Sentimento: {result.sentimento.name}"),
                            html.P(f"Confiança: {result.confianca:.2%}"),
                            html.P(f"Prioridade: {result.prioridade.name}"),
                            html.P(f"Palavras-chave: {', '.join(result.palavras_chave)}")
                        ], style={'color': alert_color, 'margin-top': '20px'}),
                        self.data.to_dict('records')
                    ]
                except Exception as e:
                    logger.error(f"Erro na análise: {str(e)}")
                    return html.Div(f"Erro: {str(e)}", style={'color': 'red'}), dash.no_update

            return dash.no_update, dash.no_update

        @self.app.callback(
            Output('sentiment-distribution', 'figure'),
            Input('analysis-data', 'data')
        )
        def update_sentiment_distribution(data):
            if not data:
                return {}
            df = pd.DataFrame(data)
            fig = px.histogram(df, x='sentimento', color='sentimento',
                               category_orders={'sentimento': ['POSITIVO', 'NEUTRO', 'NEGATIVO', 'CRITICO']},
                               title='Distribuição de Sentimentos')
            return fig

        @self.app.callback(
            Output('priority-distribution', 'figure'),
            Input('analysis-data', 'data')
        )
        def update_priority_distribution(data):
            if not data:
                return {}
            df = pd.DataFrame(data)
            fig = px.histogram(df, x='prioridade', color='prioridade',
                               category_orders={'prioridade': ['BAIXA', 'MEDIA', 'ALTA', 'URGENTE']},
                               title='Distribuição de Prioridades')
            return fig

        @self.app.callback(
            Output('time-trend', 'figure'),
            Input('analysis-data', 'data')
        )
        def update_time_trend(data):
            if not data:
                return {}
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            sentiment_counts = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'sentimento']).size().reset_index(name='count')
            fig = px.line(sentiment_counts, x='timestamp', y='count', color='sentimento',
                          category_orders={'sentimento': ['POSITIVO', 'NEUTRO', 'NEGATIVO', 'CRITICO']},
                          title='Tendência de Sentimentos ao Longo do Tempo')
            return fig

    def run(self, debug=False):
        self.app.run_server(debug=debug)
