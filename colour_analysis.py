from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uuid
import cv2
import numpy as np

# --- 1. HIRE THE FILING CLERK (SQLAlchemy Imports) ---
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# --- 2. CONNECT TO MYSQL WORKBENCH ---
DATABASE_URL = "mysql+pymysql://root:Annuasmi@127.0.0.1:3306/fashion_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 3. THE FIXED DATABASE BLUEPRINT ---
# Notice how these names now perfectly match your MySQL screenshot!
class AnalysisRecord(Base):
    __tablename__ = "user_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))           # Changed to match your DB
    skin_tone = Column(String(50))
    undertone = Column(String(50))
    palette_category = Column(String(50))    # Changed to match your DB

Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Fashion AI - Color & Skin Tone Analysis API",
    description="Backend API for processing face photos and returning personalized color recommendations.",
    version="1.0.0"
)

# Enable CORS so your frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models (The Strict Menu) ---
class ColorSwatch(BaseModel):
    name: str
    hex: str

class AnalysisResponse(BaseModel):
    analysis_id: str
    skin_tone: str
    undertone: str
    palette_category: str
    description: str
    best_colors: List[ColorSwatch]
    avoid_colors: List[ColorSwatch]
    style_tips: str

# --- Fashion Database (The Recipe Books) ---
PALETTE_DATA = {
    "Winter": {
        "colors": ["Ruby Red", "Emerald Green", "Sapphire Blue", "Pure White", "Black", "Icy Pink", "Royal Purple"],
        "dos": ["Wear high-contrast color blocks", "Opt for pure, cool-toned whites", "Choose jewel tones"],
        "donts": ["Avoid earthy, muted tones like beige or rust", "Stay away from warm oranges and yellows"]
    },
    "Summer": {
        "colors": ["Powder Blue", "Dusty Rose", "Lavender", "Mint Green", "Soft Navy", "Taupe"],
        "dos": ["Choose soft, muted, cool-toned colors", "Wear monochromatic pastel outfits"],
        "donts": ["Avoid harsh, bright neons", "Stay away from heavy black or pure white near the face"]
    },
    "Autumn": {
        "colors": ["Mustard Yellow", "Burnt Orange", "Olive Green", "Warm Brown", "Camel", "Terracotta"],
        "dos": ["Embrace rich, earthy, warm tones", "Mix and match textured fabrics in autumnal colors"],
        "donts": ["Avoid icy pastels", "Stay away from cool-toned pinks and blues"]
    },
    "Spring": {
        "colors": ["Peach", "Coral", "Golden Yellow", "Kelly Green", "Warm Aqua", "Cream"],
        "dos": ["Wear bright, warm, and clear colors", "Opt for ivory or cream instead of stark white"],
        "donts": ["Avoid dark, heavy, or muted colors", "Stay away from stark black near the face"]
    }
}

COLOR_HEX_MAP = {
    "Ruby Red": "#9B1B30", "Emerald Green": "#046307", "Sapphire Blue": "#0F52BA",
    "Pure White": "#F8F8F8", "Black": "#1C1C1C", "Icy Pink": "#F4C2C2",
    "Royal Purple": "#7B2D8B", "Powder Blue": "#B0C4DE", "Dusty Rose": "#DCAE96",
    "Lavender": "#C4A8D4", "Mint Green": "#98D8C8", "Soft Navy": "#4A5D7E",
    "Taupe": "#B5A99A", "Mustard Yellow": "#D4A017", "Burnt Orange": "#CC5500",
    "Olive Green": "#6B7B3A", "Warm Brown": "#8B5E3C", "Camel": "#C19A6B",
    "Terracotta": "#C9704A", "Peach": "#FFCBA4", "Coral": "#FF6B6B",
    "Golden Yellow": "#FFC107", "Kelly Green": "#4CBB17", "Warm Aqua": "#00B4B4",
    "Cream": "#FFFDD0"
}

SEASON_DESCRIPTIONS = {
    "Winter": "Your skin has cool, high-contrast undertones. You suit bold, pure colours and sharp contrasts that mirror your striking natural colouring.",
    "Summer": "Your skin carries soft, cool undertones with a delicate quality. Muted, dusty pastels harmonise beautifully with your gentle complexion.",
    "Autumn": "Your skin glows with warm, golden undertones. Rich earthy tones and nature-inspired hues make your complexion come alive.",
    "Spring": "Your skin has warm, clear undertones with a fresh luminosity. Bright, warm, and cheerful colours mirror your natural radiance."
}

STYLE_TIPS = {
    "Winter": "Reach for silver jewellery and high-contrast accessories. A bold red lip or jewel-toned scarf near your face will be transformative.",
    "Summer": "Choose rose gold or silver jewellery. Soft layering in tonal pastels creates an effortlessly chic look that flatters your colouring.",
    "Autumn": "Gold jewellery is your best friend. Invest in rich textures like suede, leather and wool in your palette — they elevate every outfit.",
    "Spring": "Gold and warm-toned accessories are perfect. Keep your look fresh and bright — you can carry colours others find too bold."
}

# --- THE MASTER CHEF: Real Computer Vision Analysis ---
@app.post("/api/v1/analyze-color", response_model=AnalysisResponse)
async def analyze_skin_tone(file: UploadFile = File(...)):
    
    # 1. Check the ingredients (Is it actually an image?)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # 2. Receive the raw image and let OpenCV (cv2) read the pixels
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {str(e)}")
    
    # 3. Find the face! (Convert to grayscale because it's easier for the computer to spot shadows/shapes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
   # 4. Extract the "Flavor" (The Nose)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        raw_roi = img[y + int(h * 0.45) : y + int(h * 0.65), x + int(w * 0.4) : x + int(w * 0.6)]
    else:
        height, width = img.shape[:2]
        raw_roi = img[int(height * 0.4) : int(height * 0.6), int(width * 0.4) : int(width * 0.6)]

    # Blender: smooth out pores
    blurred_roi = cv2.medianBlur(raw_roi, 5) 
    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

    # --- 5. THE SMART STRAINER ---
    h_channel, s_channel, v_channel = cv2.split(hsv_roi)
    valid_mask = (v_channel > 50) & (v_channel < 220) & (s_channel > 20)
    rgb_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2RGB)

    # --- 6. HIGH-ACCURACY CALCULATIONS ---
    if np.any(valid_mask):
        r_avg = float(np.mean(rgb_roi[:,:,0][valid_mask]))
        g_avg = float(np.mean(rgb_roi[:,:,1][valid_mask]))
        b_avg = float(np.mean(rgb_roi[:,:,2][valid_mask]))
        val = float(np.mean(v_channel[valid_mask]))
    else:
        r_avg, g_avg, b_avg = [float(x) for x in cv2.mean(rgb_roi)[:3]]
        val = float(np.mean(v_channel))

        # --- 7. APPLY THE FORMULAS (High-Precision Version) ---
    rb_diff = r_avg - b_avg
    # bg_ratio: In cool skin, Blue is much closer to Green than in warm skin
    bg_ratio = b_avg / (g_avg + 1e-6) 

    # Decision Matrix
    if rb_diff > 65 and bg_ratio < 0.85:
        undertone = "Warm"
    elif rb_diff < 52 or bg_ratio > 0.88: 
        # High Sensitivity for Cool: If Blue is strong, it's Cool
        undertone = "Cool"
    else:
        undertone = "Neutral"

    if val > 180:
        skin_tone = "Fair"
    elif val > 120:
        skin_tone = "Medium"
    else:
        skin_tone = "Deep"

    # --- 8. THE MATH: Determine the Final Season ---
    # Fixed the 'sat' bug by calculating saturation here
    _, s_channel, _ = cv2.split(hsv_roi)
    avg_sat = np.mean(s_channel)

    if undertone == "Cool":
        # Winter is high-contrast/saturated, Summer is muted/soft
        if skin_tone in ["Medium", "Deep"] or avg_sat > 40: # Saturation threshold adjusted for OpenCV
            determined_palette = "Winter" 
        else:
            determined_palette = "Summer" 
    elif undertone == "Warm":
        if skin_tone in ["Medium", "Deep"]:
            determined_palette = "Autumn" 
        else:
            determined_palette = "Spring" 
    else: # NEUTRAL
        if skin_tone == "Fair":
            determined_palette = "Summer"
        else:
            determined_palette = "Autumn"

    # 9. Plate the Food (Prepare the response for the frontend)
    ai_results = PALETTE_DATA[determined_palette]
    palette_colors = ai_results["colors"]
    
    # Pick the best colors
    mid = len(palette_colors) // 2
    best = palette_colors[:mid]
    
    # Pick colors to avoid based on their specific undertone
    if undertone == "Warm":
        avoid_palette = PALETTE_DATA["Summer"]
    elif undertone == "Cool":
        avoid_palette = PALETTE_DATA["Autumn"]
    else: # Neutral
        avoid_palette = PALETTE_DATA["Winter"] if determined_palette in ["Autumn", "Summer"] else PALETTE_DATA["Summer"]

    avoid = avoid_palette["colors"][:3]
    generated_id = str(uuid.uuid4())[:8]

    # --- SAVE TO DATABASE (Handing the fixed receipt to the Filing Clerk) ---
    db = SessionLocal() 
    try:
        new_record = AnalysisRecord(
            filename=file.filename,             # Now saves the actual image name (e.g., "photo1.png")
            skin_tone=skin_tone,
            undertone=undertone,
            palette_category=determined_palette # Matches the MySQL column!
        )
        db.add(new_record) 
        db.commit() 
        db.refresh(new_record)       
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
    finally:
        db.close() 

    response_data = AnalysisResponse(
        analysis_id=generated_id, # Generates a random receipt ID for the order
        skin_tone=skin_tone,
        undertone=undertone,
        palette_category=determined_palette,
        description=SEASON_DESCRIPTIONS[determined_palette],
        best_colors=[ColorSwatch(name=c, hex=COLOR_HEX_MAP.get(c, "#888888")) for c in best],
        avoid_colors=[ColorSwatch(name=c, hex=COLOR_HEX_MAP.get(c, "#888888")) for c in avoid],
        style_tips=STYLE_TIPS[determined_palette]
    )

    return response_data

