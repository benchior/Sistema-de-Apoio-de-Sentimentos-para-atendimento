import os
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime
import joblib
import spacy
from spacy_language_detection import LanguageDetector
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class SentimentLabel(Enum):
    POSITIVO = auto()
    NEGATIVO = auto()
    NEUTRO = auto()
    CRITICO = auto()

class PriorityLevel(Enum):
    BAIXA = auto()
    MEDIA = auto()
    ALTA = auto()
    URGENTE = auto()

@dataclass
class AnalysisResult:
    texto: str
    sentimento: SentimentLabel
    confianca: float
    prioridade: PriorityLevel
    palavras_chave: List[str]
    timestamp: datetime = datetime.now()

class SentimentAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """Inicializa o analisador de sentimentos com modelos e processadores"""
        self.nlp = spacy.load('en_core_web_md')
        self.nlp.add_pipe('language_detector', last=True)
        self.sia = SentimentIntensityAnalyzer()
        self.model = self._load_or_train_model(model_path)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.keyword_threshold = 0.3

    def _load_or_train_model(self, model_path: Optional[str]) -> Pipeline:
        """Carrega um modelo existente ou treina um novo"""
        if model_path and os.path.exists(model_path):
            logger.info(f"Carregando modelo existente de {model_path}")
            return joblib.load(model_path)

        logger.info("Treinando novo modelo...")
        X_train = ["Eu amo este produto", "Experiência terrível", "Está ok"]
        y_train = [SentimentLabel.POSITIVO, SentimentLabel.NEGATIVO, SentimentLabel.NEUTRO]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])

        pipeline.fit(X_train, y_train)
        return pipeline

    def analyze_text(self, texto: str) -> AnalysisResult:
        """Executa análise completa de sentimentos em um texto"""
        if not texto.strip():
            raise ValueError("Texto não pode ser vazio")

        doc = self.nlp(texto)
        lang = doc._.language['language']
        if lang != 'en':
            logger.warning(f"Texto não está em inglês: {lang}")

        ml_sentiment = self._ml_analyze(texto)
        rule_sentiment = self._rule_based_analyze(texto)
        final_sentiment = self._combine_results(ml_sentiment, rule_sentiment)

        palavras_chave = self._extract_keywords(doc)
        prioridade = self._determine_priority(final_sentiment, palavras_chave)

        return AnalysisResult(
            texto=texto,
            sentimento=final_sentiment,
            confianca=max(ml_sentiment[1], rule_sentiment[1]),
            prioridade=prioridade,
            palavras_chave=palavras_chave
        )

    def _ml_analyze(self, texto: str) -> Tuple[SentimentLabel, float]:
        probas = self.model.predict_proba([texto])[0]
        pred_class = self.model.predict([texto])[0]
        confianca = max(probas)
        return pred_class, confianca

    def _rule_based_analyze(self, texto: str) -> Tuple[SentimentLabel, float]:
        scores = self.sia.polarity_scores(texto)

        if scores['compound'] >= 0.05:
            return SentimentLabel.POSITIVO, scores['compound']
        elif scores['compound'] <= -0.05:
            if scores['neg'] > 0.6:
                return SentimentLabel.CRITICO, scores['neg']
            return SentimentLabel.NEGATIVO, scores['neg']
        else:
            return SentimentLabel.NEUTRO, scores['neu']

    def _combine_results(self, ml_result: Tuple, rule_result: Tuple) -> SentimentLabel:
        ml_label, ml_conf = ml_result
        rule_label, rule_conf = rule_result

        if rule_label == SentimentLabel.CRITICO and rule_conf > 0.7:
            return rule_label

        combined_conf = (ml_conf * 0.6 + rule_conf * 0.4)

        if combined_conf > 0.7:
            return ml_label
        elif combined_conf > 0.4:
            return rule_label
        else:
            return SentimentLabel.NEUTRO

    def _extract_keywords(self, doc) -> List[str]:
        palavras_chave = []
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop:
                if token.vector_norm and token.vector_norm > self.keyword_threshold:
                    palavras_chave.append(token.text)
        return list(set(palavras_chave))[:5]

    def _determine_priority(self, sentimento: SentimentLabel, palavras_chave: List[str]) -> PriorityLevel:
        palavras_prioridade = ['urgente', 'irritado', 'reembolso', 'cancelar', 'reclamação']

        if sentimento == SentimentLabel.CRITICO:
            return PriorityLevel.URGENTE
        elif sentimento == SentimentLabel.NEGATIVO:
            if any(word in palavras_prioridade for word in palavras_chave):
                return PriorityLevel.ALTA
            return PriorityLevel.MEDIA
        elif sentimento == SentimentLabel.POSITIVO:
            return PriorityLevel.BAIXA
        else:
            return PriorityLevel.BAIXA
