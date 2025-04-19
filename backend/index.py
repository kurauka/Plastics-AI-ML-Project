from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import requests
import json
import re

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TensorFlow model
MODEL = tf.keras.models.load_model("models/1.keras")
CLASS_NAMES = [
    "HDPE (High-Density Polyethylene)",
    "OTHERS",
    "PET (polyethylene terephthalate)",
    "PP (polypropylene)",
    "PVC (Polyvinyl chloride)"
]

@app.get("/ping")
async def ping():
    return {"message": "Hello from Vercel!"}

# Image loader
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))

    return {
        "class": predicted_class,
        "confidence": confidence,
        "prediction": f"<h2>Plastic Type Prediction</h2><p><strong>Class:</strong> {predicted_class}</p><p><strong>Confidence:</strong> {confidence:.2f}</p>"
    }

# Insights request body
class InsightRequest(BaseModel):
    plastic_type: str

# Clean Gemini markdown response
def clean_json_output(response_text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)
    try:
        start = response_text.index('{')
        end = response_text.rindex('}') + 1
        return response_text[start:end]
    except ValueError:
        return response_text.strip()

# Insights endpoint
@app.post("/insights")
async def get_insights(request: InsightRequest):
    prompt = f"""
    Provide a JSON object with the following structure only:
    {{
      "Plastic_name": "",
      "Common_uses": [],
      "Recycling_category": "",
      "Environmental_impact": "",
      "Alternatives": []
    }}
    The plastic type is: {request.plastic_type}
    """

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=AIzaSyDqcZHYRtufGxHuy4RGrFKe05aIUL96E6s",
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        gemini_data = response.json()

        if "candidates" not in gemini_data:
            return {"insight": "<p>Gemini API returned no candidates. Try again later.</p>"}

        raw_output = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = clean_json_output(raw_output)
        structured = json.loads(cleaned)

        html_output = f"""
        <h2>{structured.get("Plastic_name", "")}</h2>
        <h3>Common Uses</h3>
        <ul>{"".join(f"<li>{use}</li>" for use in structured.get("Common_uses", []))}</ul>
        <h3>Recycling Category</h3>
        <p>{structured.get("Recycling_category", "")}</p>
        <h3>Environmental Impact</h3>
        <p>{structured.get("Environmental_impact", "")}</p>
        <h3>Alternatives</h3>
        <ul>{"".join(f"<li>{alt}</li>" for alt in structured.get("Alternatives", []))}</ul>
        """

        return {"insight": html_output}

    except Exception as e:
        print("Error generating insight:", e)
        return {"insight": "<p>Error generating insight. Try again later.</p>"}
