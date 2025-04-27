import logging
from sentiment_analysis.analyzer import SentimentAnalyzer, PriorityLevel
from sentiment_analysis.dashboard import SentimentDashboard
from sentiment_analysis.crm import CRMIntegrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando sistema de análise de sentimentos")

    analyzer = SentimentAnalyzer()
    dashboard = SentimentDashboard(analyzer)
    crm_integrator = CRMIntegrator()

    textos_exemplo = [
        "Eu amo o seu serviço! A equipe foi extremamente prestativa.",
        "Esta é a pior experiência que já tive. Quero meu dinheiro de volta!",
        "O produto está ok, mas a entrega demorou mais do que o esperado.",
        "URGENTE: Minha conta foi hackeada! Preciso de assistência imediata!"
    ]

    for texto in textos_exemplo:
        resultado = analyzer.analyze_text(texto)
        logger.info(f"Análise: {resultado.sentimento.name} (Prioridade: {resultado.prioridade.name})")

        if resultado.prioridade in [PriorityLevel.ALTA, PriorityLevel.URGENTE]:
            crm_integrator.send_priority_alert(resultado)

    dashboard.run(debug=True)

if __name__ == "__main__":
    main()
