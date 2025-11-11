#!/usr/bin/env python3
"""
PhishBuster: Multi-modal phishing email detector with Flask backend
"""
import os
import re
import random
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app = Flask(__name__, static_folder=FRONTEND_FOLDER)
CORS(app)

# ------------------------------
# Dataset generator CLASS
# ------------------------------
class EnhancedPhishingDataset:
    def __init__(self):
        pass

    def generate_dataset(self, n_samples=1000):
        phishing_templates = [
            {"subject": "URGENT: Your {service} account will be suspended",
             "body": "We detected suspicious activity. Click here immediately to verify: {url}",
             "service": ["PayPal","Amazon","Netflix","Bank","Apple"]},
            {"subject": "Security Alert: Unusual login detected from {location}",
             "body": "Someone tried to access your account from {location}. Confirm your identity now: {url}",
             "service": ["Gmail","Facebook","Microsoft","Instagram"]},
            {"subject": "Action Required: Verify your {service} payment information",
             "body": "Your payment failed. Update your billing info ASAP or service will be suspended: {url}",
             "service": ["Netflix","Spotify","Adobe","PayPal"]},
            {"subject": "Your package delivery failed - Action needed",
             "body": "We couldn't deliver your package. Click to reschedule: {url}",
             "service": ["FedEx","UPS","USPS","DHL"]},
            {"subject": "FINAL NOTICE: Account suspension in 24 hours",
             "body": "This is your last chance to verify your account before permanent suspension. Act now: {url}",
             "service": ["Bank","PayPal","Amazon"]} 
        ]
        legitimate_templates = [
            {"subject": "Weekly team meeting reminder",
             "body": "Hi team, just a reminder about our weekly sync tomorrow at 2 PM. See you there!",
             "service": ["Company"]},
            {"subject": "Monthly newsletter - {month}",
             "body": "Check out this month's updates, articles, and company news in our latest newsletter.",
             "service": ["Newsletter"]},
            {"subject": "Invoice #{invoice_num} for your recent purchase",
             "body": "Thank you for your purchase. Please find attached invoice for your records.",
             "service": ["Company"]},
            {"subject": "Your order has shipped",
             "body": "Good news! Your order #{order_num} has been shipped and will arrive in 3-5 business days.",
             "service": ["Amazon","Store"]},
            {"subject": "Receipt for your subscription",
             "body": "Your monthly subscription payment of ${amount} has been processed. Thank you.",
             "service": ["Service"]}
        ]

        locations = ["New York", "London", "Tokyo", "Moscow", "Unknown Location"]
        months = ["January", "February", "March", "April", "May"]

        data = []
        for i in range(n_samples):
            is_phish = random.random() < 0.5
            if is_phish:
                template = random.choice(phishing_templates)
                service = random.choice(template["service"])
                location = random.choice(locations)
                subject = template["subject"].format(service=service, location=location)
                body = template["body"].format(
                    service=service,
                    location=location,
                    url=f"http://{service.lower()}-verify-{random.randint(1000,9999)}.com/login"
                )
            else:
                template = random.choice(legitimate_templates)
                service = random.choice(template["service"])
                month = random.choice(months)
                # Ensure templates that need order_num/invoice_num get those keys
                subject = template["subject"].format(
                    service=service,
                    month=month,
                    invoice_num=random.randint(10000,99999),
                    order_num=random.randint(100000,999999)
                )
                body = template["body"].format(
                    service=service,
                    amount=random.randint(10,100),
                    invoice_num=random.randint(10000,99999),
                    order_num=random.randint(100000,999999)
                )

            label = 1 if is_phish else 0
            semantic_features = self._generate_semantic_features(subject + " " + body, is_phish)
            behavioral_features = self._generate_behavioral_features(is_phish)

            data.append({
                "email_id": i,
                "subject": subject,
                "body": body,
                "full_text": subject + " " + body,
                "label": label,
                **semantic_features,
                **behavioral_features
            })
        return pd.DataFrame(data)

    def _generate_semantic_features(self, text, is_phish):
        url_count = len(re.findall(r'http[s]?://\S+', text))
        urgency_words = [
            'urgent', 'immediate', 'asap', 'now', 'quickly', 'expire',
            'suspended', 'verify', 'confirm', 'act now', 'limited time',
            'final notice', 'action required'
        ]
        urgency_score = sum(1 for w in urgency_words if w.lower() in text.lower()) / max(1, len(urgency_words))
        misspelling_score = random.uniform(0.1, 0.4) if is_phish else random.uniform(0.0, 0.05)
        return {
            "url_count": url_count,
            "urgency_score": urgency_score,
            "text_length": len(text),
            "word_count": len(text.split()),
            "misspelling_score": misspelling_score,
            "exclamation_count": text.count('!'),
            "capital_ratio": sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        }

    def _generate_behavioral_features(self, is_phish):
        if is_phish:
            return {
                'mouse_entropy': random.uniform(0.7, 1.0),
                'click_hesitation': random.uniform(5, 15),
                'scroll_anomaly': random.uniform(0.6, 1.0),
                'hover_duration': random.uniform(2, 8),
                'typing_variance': random.uniform(0.5, 1.0),
                'session_time': random.uniform(30, 300),
                'return_visits': random.randint(0, 3),
                'forward_attempts': random.randint(0, 5)
            }
        else:
            return {
                'mouse_entropy': random.uniform(0.1, 0.4),
                'click_hesitation': random.uniform(0.5, 2),
                'scroll_anomaly': random.uniform(0.0, 0.3),
                'hover_duration': random.uniform(0.2, 1.5),
                'typing_variance': random.uniform(0.0, 0.2),
                'session_time': random.uniform(5, 50),
                'return_visits': random.randint(0, 1),
                'forward_attempts': random.randint(0, 1)
            }

# ------------------------------
# Neural Network CLASS
# ------------------------------
class MultiModalAttentionNetwork(nn.Module):
    def __init__(self, semantic_dim=384, behavioral_dim=8, hidden_dim=128):
        super().__init__()
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(behavioral_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        # note: batch_first True supported earlier; kept previous interface for compatibility
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, semantic_features, behavioral_features):
        semantic_encoded = self.semantic_encoder(semantic_features)
        behavioral_encoded = self.behavioral_encoder(behavioral_features)
        semantic_attn = semantic_encoded.unsqueeze(1)
        behavioral_attn = behavioral_encoded.unsqueeze(1)
        semantic_attended, _ = self.attention(semantic_attn, behavioral_attn, behavioral_attn)
        behavioral_attended, _ = self.attention(behavioral_attn, semantic_attn, semantic_attn)
        fused = torch.cat([
            semantic_attended.squeeze(1), behavioral_attended.squeeze(1)
        ], dim=1)
        output = torch.sigmoid(self.fusion_layer(fused))
        return output

# ------------------------------
# Feature Extractor CLASS
# ------------------------------
class AdvancedFeatureExtractor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.sentence_model = SentenceTransformer(model_name)

    def semantic(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=False)
        return torch.FloatTensor(embeddings)

    def behavioral(self, df):
        cols = [
            'mouse_entropy', 'click_hesitation', 'scroll_anomaly', 'hover_duration',
            'typing_variance', 'session_time', 'return_visits', 'forward_attempts'
        ]
        arr = df[cols].values if isinstance(df, pd.DataFrame) else np.asarray(df)
        return torch.FloatTensor(arr)

# ------------------------------
# Enhanced Risk Assessment FUNCTION
# ------------------------------
def advanced_risk_assessment(text):
    """
    Improved risk assessment combining rich patterns and context.
    Returns: risk_score (0-100), category ('Safe', 'Suspicious', 'Fraudulent')
    """
    text_lower = text.lower()
    critical_keywords = [
        'urgent', 'verify account', 'suspended', 'permanently blocked', 'blocked',
        'immediately', 'click the link', 'click here',
        'limited time', 'restore access', 'confirm your payment', 'confirm your payment details',
        'security alert', 'permanent suspension', 'failure to respond', 'action required',
        'compliance team', 'cvv', 'billing address', 'login', 'card number', 'password'
    ]
    suspicious_patterns = [
        'click', 'update', 'confirm', 'verify', 'login', 'payment', 'billing', 'address', 'restore',
        'security', 'account', 'risk', 'activity'
    ]
    url_matches = re.findall(r'http[s]?://[^\s<>"\']+', text_lower)
    domain_score = 0
    for url in url_matches:
        if 'paypal' in url and (('verify' in url) or ('security' in url) or ('confirm' in url)):
            domain_score += 20
        if any(s in url for s in ['login', 'billing', 'account', 'secure']):
            domain_score += 8
        if url.endswith('.com/confirm') or url.endswith('.com/login'):
            domain_score += 5

    critical_score = sum(15 for keyword in critical_keywords if keyword in text_lower)
    suspicious_score = sum(5 for pat in suspicious_patterns if pat in text_lower)
    urgency_score = text_lower.count('immediately') * 8 + text_lower.count('now') * 4 + text_lower.count('failure') * 5
    exclamation_score = min(3, text.count('!')) * 2
    caps_score = sum(1 for w in text.split() if w.isupper()) * 3

    score = critical_score + suspicious_score + urgency_score + exclamation_score + caps_score + domain_score
    score = min(100, score)
    if score >= 60:
        category = 'Fraudulent'
        confidence = min(97, 77 + (score - 60) * 0.8)
    elif score >= 35:
        category = 'Suspicious'
        confidence = min(85, 62 + (score - 35) * 0.7)
    else:
        category = 'Safe'
        confidence = max(88, 97 - score * 0.2)
    return score, category, round(confidence, 1)

# ------------------------------
# Train model and features
# ------------------------------
print("Preparing dataset and training model...")
dataset_gen = EnhancedPhishingDataset()
df = dataset_gen.generate_dataset(500)

feature_extractor = AdvancedFeatureExtractor()
X_semantic = feature_extractor.semantic(df['full_text'].tolist())
X_behavioral = feature_extractor.behavioral(df)
y = torch.FloatTensor(df['label'].values).unsqueeze(1)

X_train_s, X_test_s, X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_semantic, X_behavioral, y, test_size=0.2, random_state=42, stratify=y
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalAttentionNetwork(
    semantic_dim=X_semantic.shape[1],
    behavioral_dim=X_behavioral.shape[1]
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) #lr=learning rate

for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_s.to(device), X_train_b.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_test_s.to(device), X_test_b.to(device))
        val_preds = (val_out > 0.5).float()
        val_acc = (val_preds == y_test.to(device)).float().mean().item()
    print(f"Epoch {epoch + 1}/3 | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

# ------------------------------
# Flask routes
# ------------------------------
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/style.css')
def serve_css():
    return send_from_directory(FRONTEND_FOLDER, 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory(FRONTEND_FOLDER, 'script.js')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'status': 'error', 'message': 'No text provided'}), 400

    # Core risk assessment + ML prob
    risk_score, category, confidence = advanced_risk_assessment(text)
    semantic = feature_extractor.semantic([text]).to(device)
    behavioral = torch.zeros((1, X_behavioral.shape[1]), dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        ml_prob = float(model(semantic, behavioral).item())

    # Adjust confidence slightly based on ML signal
    if (category == 'Fraudulent' and ml_prob > 0.6) or (category == 'Safe' and ml_prob < 0.3):
        confidence += 4
    elif (category == 'Fraudulent' and ml_prob < 0.3 and risk_score < 90) or (category == 'Safe' and ml_prob > 0.7):
        confidence -= 12

    # ---------- NEW: Additional Analysis ----------
    words = [w for w in re.split(r'\s+', text) if w]
    word_count = len(words)
    suspicious_keywords = [
        'urgent', 'verify', 'account', 'password', 'click', 'login', 'security',
        'payment', 'billing', 'confirm', 'limited', 'action', 'immediately', 'verify'
    ]
    keyword_hits = [w for w in words if any(k in w.lower() for k in suspicious_keywords)]
    keyword_count = len(keyword_hits)

    # URL analysis
    url_matches = re.findall(r'http[s]?://[^\s<>"\']+', text)
    url_count = len(url_matches)
    malicious_url_count = 0
    domains = []
    for u in url_matches:
        domains.append(re.sub(r'https?://', '', u).split('/')[0].lower())
        # simple heuristic: urls containing 'verify' or suspicious TLD patterns considered more risky
        if 'verify' in u or '-verify' in u or u.count('.') >= 3:
            malicious_url_count += 1

    # domain trust (simulated): if domain contains brand keywords -> medium trust; if 'secure' or weird tokens -> low trust
    domain_trust_score = 80  # default
    if domains:
        d = domains[0]
        if any(b in d for b in ['paypal','amazon','bank','apple']):
            domain_trust_score = 40  # impersonation risk
        if any(tok in d for tok in ['verify','secure','account','login','update']):
            domain_trust_score = 25

    # sender reputation (simulated from presence of brand names or suspicious words)
    sender_reputation = "Unknown"
    if any(b in text.lower() for b in ['paypal', 'bank', 'amazon', 'apple', 'netflix']):
        sender_reputation = "Impersonation Risk"
    elif keyword_count > max(2, 0.02 * word_count):
        sender_reputation = "Low"
    else:
        sender_reputation = "Normal"

    # urgency tone
    if 'immediately' in text.lower() or 'urgent' in text.lower() or text.count('!') >= 2:
        urgency_tone = "High"
    elif 'soon' in text.lower() or 'now' in text.lower():
        urgency_tone = "Medium"
    else:
        urgency_tone = "Low"

    # sensitive request detection
    sensitive_phrases = ['password', 'card number', 'cvv', 'otp', 'one-time', 'ssn', 'social security']
    sensitive_request = any(p in text.lower() for p in sensitive_phrases)

    # language quality score (simple heuristic using misspelling_score if present)
    # estimate misspellings via words with non-alpha or many digits (very rough)
    non_alpha_words = sum(1 for w in words if re.search(r'[^A-Za-z\-@./:]', w))
    language_score = max(20, 100 - (non_alpha_words * 3) - int((keyword_count / max(1, word_count)) * 100))

    # link mismatch detection (we can't parse HTML anchors in plain text, so simulated)
    link_mismatch = False
    if url_count > 0 and any(tok in text.lower() for tok in ['click here', 'visit', 'link below']):
        link_mismatch = True

    # attachment risk (simulated — you can extend to inspect real attachments when integrated)
    attachment_risk = "None"
    if re.search(r'\.(exe|zip|scr|bat|js)\b', text.lower()):
        attachment_risk = "High"
    elif re.search(r'\.(pdf|docx|xlsx)\b', text.lower()):
        attachment_risk = "Medium"

    # screen time simulated (if you ever collect real behavioral data, replace this)
    screen_time = round(random.uniform(5, 240), 2)

    # keyword density
    keyword_density = round((keyword_count / max(1, word_count)) * 100, 2)

    # phish risk index (map risk_score into a 0-100 index, already 0-100)
    phish_risk_index = risk_score

    analysis_summary = (
    f"• The email contains {word_count} words.\n"
    f"• It includes {keyword_count} suspicious keywords.\n"
    f"• There are {url_count} link(s), out of which {malicious_url_count} appear suspicious.\n"
    f"• Calculated Risk Index: {phish_risk_index}.\n"
    f"• Final Classification: {category} (Confidence: {confidence}%)."
)



    behavioral_data = {
        "word_count": word_count,
        "suspicious_keyword_count": keyword_count,
        "keyword_density(%)": keyword_density,
        "url_count": url_count,
        "malicious_url_count": malicious_url_count,
        "domains": domains,
        "sender_reputation": sender_reputation,
        "domain_trust_score": domain_trust_score,
        "urgency_tone": urgency_tone,
        "sensitive_request": sensitive_request,
        "language_score": language_score,
        "link_mismatch": link_mismatch,
        "attachment_risk": attachment_risk,
        "screen_time_sec": screen_time,
        "ml_probability(%)": round(ml_prob * 100, 1),
        "risk_score": risk_score,
        "phish_risk_index": phish_risk_index,
        "category": category,
        "confidence": round(min(max(confidence, 0), 99.9), 1),
        "analysis_summary": analysis_summary
    }
    # ----------------------------------------------

    return jsonify({
        'prediction': category,
        'confidence': round(min(max(confidence, 0), 99.9), 1),
        'risk_score': risk_score,
        'ml_probability': round(ml_prob * 100, 1),
        'behavioral': behavioral_data  # <-- send to frontend
    })


if __name__ == '__main__':
    print("Starting Flask PhishBuster app...")
    app.run(host='0.0.0.0', port=5000, debug=True)
