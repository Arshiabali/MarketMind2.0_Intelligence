"""MarketMind Intelligence V2 — standalone Gradio deployment.

Customer sentiment and aspects use trained review models and semantic similarity.
Demand and inventory use simulated business data; outputs are illustrative.
"""
import json
import os
import re
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

metadata = read_json(MODELS / "deployment_metadata.json")
v2_config = read_json(MODELS / "v2_nlp_config.json")
DEMAND_FEATURES = metadata["demand_features"]
business_latest = pd.read_csv(DATA / "business_latest.csv")
business_latest["product_id"] = business_latest["product_id"].astype(str)

best_model = joblib.load(MODELS / "sentiment_model.pkl")
tfidf = joblib.load(MODELS / "tfidf_vectorizer.pkl")
loaded_demand_model = joblib.load(MODELS / "demand_prediction_model.pkl")

# Load original theme models as part of the exported V1 assets.
# Customer Intelligence V2 uses semantic aspect matching instead of theme clusters.
theme_vectorizer = joblib.load(MODELS / "theme_vectorizer.pkl")
kmeans_model = joblib.load(MODELS / "kmeans_model.pkl")

semantic_model = SentenceTransformer(v2_config["semantic_model"])
ASPECTS = v2_config["aspect_descriptions"]
aspect_embeddings = {
    aspect: semantic_model.encode(
        descriptions, convert_to_tensor=True, normalize_embeddings=True
    )
    for aspect, descriptions in ASPECTS.items()
}

def split_review(review):
    parts = re.split(
        r"[.!?;]+|\bbut\b|\bhowever\b",
        review,
        flags=re.IGNORECASE
    )
    return [p.strip() for p in parts if p.strip()]

def predict_text_sentiment(text):
    vector = tfidf.transform([text])
    sentiment = str(best_model.predict(vector)[0])
    probabilities = best_model.predict_proba(vector)[0]
    confidence = float(np.max(probabilities))
    return sentiment, confidence

def detect_aspects(review, threshold=None):
    if threshold is None:
        threshold = float(v2_config["aspect_similarity_threshold"])
    results = []

    for sentence in split_review(review):
        embedding = semantic_model.encode(
            sentence,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        sentiment, confidence = predict_text_sentiment(sentence)

        for aspect, reference_embeddings in aspect_embeddings.items():
            score = util.cos_sim(
                embedding,
                reference_embeddings
            ).max().item()

            if score >= threshold:
                results.append({
                    "Aspect": aspect,
                    "Sentiment": sentiment,
                    "Similarity": round(float(score), 3),
                    "Confidence": round(confidence, 3),
                    "Evidence": sentence
                })

    return results

def check_rating_mismatch(sentiment, rating, confidence):
    if rating is None:
        return "Not checked"

    if confidence < float(v2_config["mismatch_confidence_threshold"]):
        return "Uncertain — manual review recommended"

    if rating <= 2 and sentiment == "Positive":
        return "Potential mismatch"

    if rating >= 4 and sentiment == "Negative":
        return "Potential mismatch"

    if rating == 3 or sentiment == "Neutral":
        return "Ambiguous — manual review recommended"

    return "No obvious mismatch"

def analyze_review_v2(review, rating=None):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please enter a review.")

    sentiment, confidence = predict_text_sentiment(review)
    aspects = detect_aspects(review)

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "aspects": aspects,
        "rating_consistency": check_rating_mismatch(
            sentiment, rating, confidence
        )
    }




def product_features(product_id, price, rating, negative_pct,
                     lead_time_days, promotion, current_stock=None):
    matching = business_latest.loc[
        business_latest["product_id"] == str(product_id)
    ]
    if matching.empty:
        raise ValueError("Select a valid product ID.")
    price, rating, negative_pct, lead_time_days = map(
        float, (price, rating, negative_pct, lead_time_days)
    )
    if price < 0 or not 1 <= rating <= 5 or not 0 <= negative_pct <= 100 or lead_time_days < 0:
        raise ValueError("Use price >= 0, rating 1–5, negative feedback 0–100%, lead time >= 0.")
    features = matching.iloc[-1][DEMAND_FEATURES].to_dict()
    features.update(
        price=price, rating=rating, negative_pct=negative_pct,
        lead_time_days=lead_time_days, promotion=int(promotion)
    )
    if current_stock is not None:
        if float(current_stock) < 0:
            raise ValueError("Stock cannot be negative.")
        features["current_stock"] = float(current_stock)
    return features


def predict_future_demand(
    business_features
):

    input_df = pd.DataFrame(
        [business_features]
    )

    missing_features = [
        feature
        for feature in DEMAND_FEATURES
        if feature not in input_df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing required features: "
            + str(missing_features)
        )

    input_df = input_df[
        DEMAND_FEATURES
    ]

    prediction = (
        loaded_demand_model
        .predict(input_df)[0]
    )

    return max(
        0,
        float(prediction)
    )
def generate_inventory_recommendation(
    business_features
):

    predicted_demand = (
        predict_future_demand(
            business_features
        )
    )

    daily_demand = (
        predicted_demand / 7
    )

    lead_time = (
        business_features[
            "lead_time_days"
        ]
    )

    current_stock = (
        business_features[
            "current_stock"
        ]
    )

    sales_std = (
        business_features.get(
            "sales_std_7",
            0
        )
    )

    safety_stock = (
        sales_std
        * np.sqrt(lead_time)
        * 1.65
    )

    lead_time_demand = (
        daily_demand
        * lead_time
    )

    reorder_point = (
        lead_time_demand
        + safety_stock
    )

    recommended_order = max(
        0,
        np.ceil(
            predicted_demand
            + safety_stock
            - current_stock
        )
    )

    if current_stock <= (
        0.5 * reorder_point
    ):

        status = "URGENT RESTOCK"

    elif current_stock < reorder_point:

        status = "RESTOCK SOON"

    else:

        status = "STOCK HEALTHY"

    return {

        "predicted_7_day_demand":
            round(
                predicted_demand,
                2
            ),

        "daily_demand":
            round(
                daily_demand,
                2
            ),

        "lead_time_demand":
            round(
                lead_time_demand,
                2
            ),

        "safety_stock":
            round(
                safety_stock,
                2
            ),

        "reorder_point":
            round(
                reorder_point,
                2
            ),

        "current_stock":
            current_stock,

        "recommended_order_quantity":
            int(recommended_order),

        "stock_status":
            status
    }
def website_customer_analysis_v2(review, rating):
    try:
        rating_value = (
            None if rating == "Not provided"
            else int(rating)
        )

        result = analyze_review_v2(review, rating_value)

        aspect_rows = [
            [
                item["Aspect"],
                item["Sentiment"],
                item["Similarity"],
                item["Evidence"]
            ]
            for item in result["aspects"]
        ]

        explanation = (
            "Review-specific aspect matches are shown "
            "with their supporting text. "
            "Model probabilities are not calibrated."
        )

        return (
            result["sentiment"],
            f'{result["confidence"]:.1%}',
            result["rating_consistency"],
            aspect_rows,
            explanation
        )

    except Exception as error:
        return "Error", "", "Not checked", [], str(error)


def website_demand_prediction(product_id, price, rating, negative_pct,
                              lead_time_days, promotion):
    try:
        features = product_features(
            product_id, price, rating, negative_pct, lead_time_days, promotion
        )
        forecast = predict_future_demand(features)
        return (
            f"{forecast:.1f} units / next 7 days",
            "Forecast based on simulated historical sales and product features."
        )
    except Exception as exc:
        return "Error", str(exc)


def website_inventory_recommendation(product_id, current_stock, price,
                                     rating, negative_pct, lead_time_days, promotion):
    try:
        features = product_features(
            product_id, price, rating, negative_pct, lead_time_days,
            promotion, current_stock
        )
        rec = generate_inventory_recommendation(features)
        table = pd.DataFrame([
            {"Metric": key.replace("_", " ").title(), "Value": value}
            for key, value in rec.items()
        ])
        return rec["stock_status"], table
    except Exception as exc:
        return "Error", pd.DataFrame([{"Metric": "Error", "Value": str(exc)}])


def create_dashboard():
    stats = metadata["review_stats"]
    return f"""# MarketMind Business Intelligence Dashboard

## Customer Intelligence
**Total Reviews:** {stats['total_reviews']:,}

**Products Analyzed:** {stats['total_products']:,}

### Sentiment Distribution
- Positive Reviews: {stats['positive']:,}
- Neutral Reviews: {stats['neutral']:,}
- Negative Reviews: {stats['negative']:,}

---

## Business Intelligence
Customer Feedback → Sentiment → Review-Specific Aspects → Demand Prediction → Inventory Intelligence → Restock Recommendation

**Important:** Customer feedback uses an Amazon Beauty review dataset. Demand and inventory use simulated business data and are demonstrations, not validated real-world forecasts.
"""


product_choices = sorted(business_latest['product_id'].unique().tolist())
if not product_choices:
    raise RuntimeError("No products in business_latest.csv")
with gr.Blocks(title='MarketMind | Business Intelligence') as marketmind_app:
    gr.Markdown('# 📊 MarketMind — AI Business Intelligence')
    gr.Markdown('Customer sentiment, review themes, 7-day demand forecasts and inventory recommendations. **Demand and inventory data are simulated demonstrations.**')
    with gr.Tab('📊 Dashboard'):
        dashboard_output=gr.Markdown(create_dashboard())
        gr.Button('Refresh Dashboard').click(create_dashboard,outputs=dashboard_output)

    with gr.Tab('💬 Customer Intelligence V2'):
        review_input = gr.Textbox(
            label='Customer Review',
            lines=4,
            placeholder='Great quality but terrible delivery.'
        )

        rating_input = gr.Dropdown(
            choices=[
                'Not provided', '1', '2', '3', '4', '5'
            ],
            value='Not provided',
            label='Star Rating (Optional)'
        )

        analyze_btn = gr.Button(
            'Analyze Review',
            variant='primary'
        )

        sentiment_output = gr.Textbox(
            label='Overall Sentiment'
        )
        confidence_output = gr.Textbox(
            label='Model Probability (Uncalibrated)'
        )
        mismatch_output = gr.Textbox(
            label='Rating Consistency'
        )

        aspect_output = gr.Dataframe(
            headers=[
                'Aspect', 'Sentiment',
                'Similarity', 'Evidence'
            ],
            datatype=['str', 'str', 'number', 'str'],
            interactive=False,
            label='Review-Specific Aspects'
        )

        explanation_output = gr.Textbox(
            label='Interpretation'
        )

        analyze_btn.click(
            fn=website_customer_analysis_v2,
            inputs=[review_input, rating_input],
            outputs=[
                sentiment_output,
                confidence_output,
                mismatch_output,
                aspect_output,
                explanation_output
            ]
        )

    with gr.Tab('📈 Demand Intelligence'):
        demand_product=gr.Dropdown(choices=product_choices,value=product_choices[0],label='Product ID')
        with gr.Row():
            demand_price=gr.Number(value=100,label='Price')
            demand_rating=gr.Number(value=4,label='Rating (1–5)')
            demand_negative=gr.Number(value=10,label='Negative Feedback (%)')
        with gr.Row():
            demand_lead=gr.Number(value=7,label='Lead Time (days)')
            demand_promo=gr.Dropdown(choices=[0,1],value=0,label='Promotion Active?')
        demand_result=gr.Textbox(label='7-Day Demand Prediction')
        demand_explanation=gr.Textbox(label='Explanation')
        gr.Button('Predict Demand',variant='primary').click(website_demand_prediction,inputs=[demand_product,demand_price,demand_rating,demand_negative,demand_lead,demand_promo],outputs=[demand_result,demand_explanation])
    with gr.Tab('📦 Inventory Intelligence'):
        inv_product=gr.Dropdown(choices=product_choices,value=product_choices[0],label='Product ID')
        with gr.Row():
            inv_stock=gr.Number(value=100,label='Current Stock')
            inv_price=gr.Number(value=100,label='Price')
            inv_rating=gr.Number(value=4,label='Rating (1–5)')
        with gr.Row():
            inv_negative=gr.Number(value=10,label='Negative Feedback (%)')
            inv_lead=gr.Number(value=7,label='Lead Time (days)')
            inv_promo=gr.Dropdown(choices=[0,1],value=0,label='Promotion Active?')
        inv_status=gr.Textbox(label='Stock Status')
        inv_table=gr.Dataframe(label='Inventory Recommendation',interactive=False)
        gr.Button('Generate Restock Recommendation',variant='primary').click(website_inventory_recommendation,inputs=[inv_product,inv_stock,inv_price,inv_rating,inv_negative,inv_lead,inv_promo],outputs=[inv_status,inv_table])

if __name__ == "__main__":
    marketmind_app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True
    )
