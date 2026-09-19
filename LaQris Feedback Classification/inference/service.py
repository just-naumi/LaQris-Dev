"""
Service Wrapper Klasifikasi Feedback untuk Integrasi Backend LaQris
"""

import os
from .predict import load_predictor, predict_feedback

class QRISFeedbackClassifierService:
    def __init__(self, model_path=None):
        self.tokenizer, self.model, self.id2label = load_predictor(model_path)

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "category": "TRANSAKSI_SUKSES_NORMAL",
                "confidence": 1.0,
                "severity": "LOW",
                "recommended_action": "Teks kosong."
            }
        return predict_feedback(text.strip(), self.tokenizer, self.model, self.id2label)

# Singleton instance
_service = None

def get_feedback_classifier_service():
    global _service
    if _service is None:
        _service = QRISFeedbackClassifierService()
    return _service
