import os
import json
import logging
import requests
from sentiment_analysis.analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class CRMIntegrator:
    def __init__(self):
        self.api_url = os.getenv('CRM_API_URL')
        self.api_key = os.getenv('CRM_API_KEY')

    def send_priority_alert(self, analysis_result: AnalysisResult) -> bool:
        if not self.api_url or not self.api_key:
            logger.warning("Credenciais da API não configuradas")
            return False

        payload = {
            'texto': analysis_result.texto,
            'sentimento': analysis_result.sentimento.name,
            'prioridade': analysis_result.prioridade.name,
            'palavras_chave': analysis_result.palavras_chave,
            'timestamp': analysis_result.timestamp.isoformat()
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(
                f"{self.api_url}/alerts",
                data=json.dumps(payload),
                headers=headers
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha na integração com CRM: {str(e)}")
            return False
