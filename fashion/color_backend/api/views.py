import uuid
import cv2
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render

# --- Premium Fashion Data Engine Maps ---
PALETTE_DATA = {
    "Winter": {
        "title": "Cool Tone (Blue-Based)",
        "season": "Winter",
        "cheesy": "Are you sculpted from pure glacier ice? Because you're giving absolute frostbite to anyone looking too close! Icy blue hues and raspberry coordinates are your personal runway crown jewels.",
        "signature_name": "Icy Blue-Based Crystal",
        "signature_desc": "Your physical parameters display deep cooler blood matrices and pink/blue light dispersion profiles that thrive in high contrast.",
        "metal_title": "Sterling Silver & Platinum",
        "metal_desc": "Your skin's underlying pigment parameters perfectly capture and reflect cool silver waves like liquid moonlight. Steer clear of brassy, overly warm golds.",
        "metal_glow": "#a5f3fc",
        "hair_title": "Icy Platinum & Cool Espresso",
        "hair_desc": "Ashy tones are your absolute crown royalty. Embrace high-fashion platinum blonde, cool ash browns, dark espresso, or jet obsidian black.",
        "hair_glow": "#be185d",
        "metal_swatches": [{"name": "Moonlight Silver", "hex": "#cbd5e1"}, {"name": "White Gold", "hex": "#f1f5f9"}, {"name": "Prism Platinum", "hex": "#e2e8f0"}],
        "hair_swatches": [{"name": "Icy Platinum", "hex": "#e2e8f0"}, {"name": "Ash Brown", "hex": "#5c524d"}, {"name": "Obsidian Black", "hex": "#0a0a0a"}, {"name": "Cool Espresso", "hex": "#27211e"}],
        "glow_colors": [
            {"name": "Icy Celestial Blue", "hex": "#a5f3fc", "desc": "Reflects immaculate clear, cool slate-like brightness near your face."},
            {"name": "Raspberry Seduction", "hex": "#be185d", "desc": "Brings out the vivid healthy rosy glow of your natural circulation."},
            {"name": "Royal Lavender", "hex": "#c084fc", "desc": "Neutralizes sallow reflections and highlights clean jaw symmetry."},
            {"name": "Vibrant Emerald", "hex": "#047857", "desc": "Creates dramatic high-contrast perimeter that elevates the iris focus."},
            {"name": "Orchid Pink", "hex": "#db2777", "desc": "An energetic cool pink that highlights cheek contour lines beautifully."},
            {"name": "Midnight Sapphire", "hex": "#1e3a8a", "desc": "A stately, powerful deep blue that frames skin with rich intensity."},
            {"name": "Silver Mist", "hex": "#94a3b8", "desc": "A sleek, metallic-reflecting grey that complements icy eye hues."},
            {"name": "Cool Cranberry", "hex": "#9d174d", "desc": "Deep wine red that accentuates lip definition and sharp features."}
        ],
        "universal_colors": [{"name": "Classic Navy", "hex": "#1e3a8a"}, {"name": "Deep Charcoal", "hex": "#374151"}, {"name": "Heather Slate", "hex": "#6b7280"}],
        "avoid_colors": [
            {"name": "Vibrant Tangerine Orange", "hex": "#f97316", "desc": "Clashes with cool proteins, casting yellow sallow shades onto your chin."},
            {"name": "Mustard Gold", "hex": "#eab308", "desc": "Accentuates under-eye dark circles and fatigue shadows instantly."},
            {"name": "Cinnamon Sand", "hex": "#7c2d12", "desc": "Can wash out and completely dull natural crown hair highlights."}
        ]
    },
    "Summer": {
        "title": "Cool Tone (Blue-Based)",
        "season": "Summer",
        "cheesy": "Soft, cool, and beautifully delicate. Muted twilight lavenders and powder blue palettes establish an effortless elegance across your complexion.",
        "signature_name": "Muted Twilight Slate",
        "signature_desc": "Your skin displays gentle cool undertones with soft clarity, matching muted and delicate shade intervals.",
        "metal_title": "Sterling Silver & Rose Gold",
        "metal_desc": "Soft brushed metals mirror your delicate cool depth safely. Avoid high-shine heavy golds.",
        "metal_glow": "#b0c4de",
        "hair_title": "Ash Blonde & Cool Brown",
        "hair_desc": "Muted tones provide absolute balance. Stick with ash blonde, mushroom brown, or light platinum highlights.",
        "hair_glow": "#dcae96",
        "metal_swatches": [{"name": "Brushed Silver", "hex": "#e2e8f0"}, {"name": "Soft Rose Gold", "hex": "#fecdd3"}],
        "hair_swatches": [{"name": "Mushroom Brown", "hex": "#8a7e72"}, {"name": "Ash Blonde", "hex": "#cbd5e1"}],
        "glow_colors": [
            {"name": "Powder Blue", "hex": "#b0c4de", "desc": "Complements your soft-focus parameters flawlessly."},
            {"name": "Dusty Rose", "hex": "#dcae96", "desc": "Lifts check coloration without aggressive saturation spikes."},
            {"name": "Soft Lavender", "hex": "#c4a8d4", "desc": "Establishes calm, high-end visual tracking balances."},
            {"name": "Mint Green", "hex": "#98d8c8", "desc": "Softly neutralizes superficial redness across the skin matrix."},
            {"name": "Soft Navy", "hex": "#4a5d7e", "desc": "A classic framing baseline tone offering soft luxury shadows."},
            {"name": "Muted Taupe", "hex": "#b5a99a", "desc": "Subtle grounding neutral optimal for casual everyday tailoring."}
        ],
        "universal_colors": [{"name": "Classic Navy", "hex": "#1e3a8a"}, {"name": "Slate Mist", "hex": "#4b5563"}],
        "avoid_colors": [
            {"name": "Neon Tangerine", "hex": "#ff6b6b", "desc": "Overwhelms your delicate soft coloration profile profile instantly."},
            {"name": "Stark Heavy Black", "hex": "#000000", "desc": "Creates overly severe shadows near your eye parameters."}
        ]
    },
    "Autumn": {
        "title": "Warm Tone (Yellow-Based)",
        "season": "Autumn",
        "cheesy": "Did you just walk out of a golden hour filter? Warm terracotta and rich mossy forest shades are practically bowing down to your rich, glowing presence!",
        "signature_name": "Golden Earth Velvet",
        "signature_desc": "Your chromatic profiles contain heavy warm golden matrices that thrive under organic, nature-inspired tonal selections.",
        "metal_title": "18K Yellow Gold & Matte Bronze",
        "metal_desc": "The rich, reflective properties of yellow gold harmonize beautifully with your warm cellular proteins, maximizing your inner radiance.",
        "metal_glow": "#fb923c",
        "hair_title": "Rich Chestnut & Copper Auburn",
        "hair_desc": "Warm dark tones elevate your hair framing. Look into rich mahogany, copper, chestnut brown, or golden black roots.",
        "hair_glow": "#b45309",
        "metal_swatches": [{"name": "18K Gold", "hex": "#fbbf24"}, {"name": "Warm Bronze", "hex": "#92400e"}],
        "hair_swatches": [{"name": "Rich Chestnut", "hex": "#451a03"}, {"name": "Copper Auburn", "hex": "#7c2d12"}],
        "glow_colors": [
            {"name": "Terracotta Sunset", "hex": "#c2410c", "desc": "Accentuates your natural golden-peach matrix and adds depth."},
            {"name": "Rich Forest Moss", "hex": "#15803d", "desc": "Sets a beautifully organic, grounded perimeter around your cheek structures."},
            {"name": "Toasted Mustard", "hex": "#b45309", "desc": "Flashes golden hour vibes onto your facial center, highlighting skin sheen."},
            {"name": "Desert Amber", "hex": "#d97706", "desc": "A glowing, rich golden-yellow that enhances gold undertones."},
            {"name": "Warm Cocoa", "hex": "#78350f", "desc": "A rich chocolatey neutral that serves as a stunning structural frame."},
            {"name": "Olive Branch", "hex": "#65a30d", "desc": "Balances minor face flush while amplifying healthy organic skin metrics."}
        ],
        "universal_colors": [{"name": "Warm Off-White", "hex": "#fafaf9"}, {"name": "Rich Camel", "hex": "#ca8a04"}],
        "avoid_colors": [
            {"name": "Icy Slate Grey", "hex": "#9ca3af", "desc": "Drains your healthy solar warmth, making you look artificially tired."},
            {"name": "Neon Pink", "hex": "#f43f5e", "desc": "Clashes aggressively against the natural warmth of your baseline pigment."}
        ]
    },
    "Spring": {
        "title": "Warm Tone (Yellow-Based)",
        "season": "Spring",
        "cheesy": "Luminous, bright, and filled with golden energy. Vibrant peach corals and warm marigolds make your clear natural complexion glow instantly.",
        "signature_name": "Luminous Sun Blossom",
        "signature_desc": "High clarity meets warm base metrics. Your skin exhibits a lively peach-gold luminosity that pops under bright tones.",
        "metal_title": "Bright Yellow Gold & Champagne Gold",
        "metal_desc": "Polished, shiny golds enhance the fresh, clear brightness of your skin. Silver can sometimes appear too cool and flat.",
        "metal_glow": "#eab308",
        "hair_title": "Golden Blonde & Honey Brown",
        "hair_desc": "Keep your hair highlights bright and sunny. Honey blonde, golden brown, or warm camel streaks elevate your framing profile.",
        "hair_glow": "#ca8a04",
        "metal_swatches": [{"name": "Champagne Gold", "hex": "#fde047"}, {"name": "Bright Gold", "hex": "#eab308"}],
        "hair_swatches": [{"name": "Honey Blonde", "hex": "#fef08a"}, {"name": "Golden Brown", "hex": "#b45309"}],
        "glow_colors": [
            {"name": "Premium Peach Coral", "hex": "#fb923c", "desc": "Unlocks rich warm rosiness, making you look awake and refreshed instantly."},
            {"name": "Marigold Bloom", "hex": "#f59e0b", "desc": "Vibrant warm yellow that makes natural eye flecks pop with energy."},
            {"name": "Warm Aqua", "hex": "#06b6d4", "desc": "Bright tropical blue-green that creates a stunning high-end framing aura."},
            {"name": "Chiffon Cream", "hex": "#fef9c3", "desc": "Soft, gentle golden-cream that doesn't overpower neutral value scales."}
        ],
        "universal_colors": [{"name": "Universal Off-White", "hex": "#f5f5f4"}, {"name": "Warm Taupe", "hex": "#ca8a04"}],
        "avoid_colors": [
            {"name": "Heavy Obsidian Black", "hex": "#000000", "desc": "Completely overpowers your delicate, bright natural features."},
            {"name": "Dull Ash Grey", "hex": "#6b7280", "desc": "Mutes your natural solar radiance, casting a flat, washed-out finish."}
        ]
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

@api_view(['POST'])
def analyze_color_api(request):
    if 'file' not in request.FILES:
        return Response({"detail": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)
    
    uploaded_file = request.FILES['file']
    
    try:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return Response({"detail": "Corrupted or invalid image structure."}, status=status.HTTP_400_BAD_REQUEST)
    except Exception:
        return Response({"detail": "Error reading image upload stream."}, status=status.HTTP_400_BAD_REQUEST)
    
    # --- 1. OpenCV Computer Vision Matrix Processing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        raw_roi = img[y + int(h * 0.45) : y + int(h * 0.65), x + int(w * 0.4) : x + int(w * 0.6)]
    else:
        height, width = img.shape[:2]
        raw_roi = img[int(height * 0.4) : int(height * 0.6), int(width * 0.4) : int(width * 0.6)]

    blurred_roi = cv2.medianBlur(raw_roi, 5) 
    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_roi)
    valid_mask = (v_channel > 50) & (v_channel < 220) & (s_channel > 20)
    rgb_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2RGB)

    if np.any(valid_mask):
        r_avg = float(np.mean(rgb_roi[:,:,0][valid_mask]))
        g_avg = float(np.mean(rgb_roi[:,:,1][valid_mask]))
        b_avg = float(np.mean(rgb_roi[:,:,2][valid_mask]))
        val = float(np.mean(v_channel[valid_mask]))
    else:
        r_avg, g_avg, b_avg = [float(x) for x in cv2.mean(rgb_roi)[:3]]
        val = float(np.mean(v_channel))

    # --- 2. HIGH-PRECISION SELECTION FORMULAS ---
    rb_diff = r_avg - b_avg
    bg_ratio = b_avg / (g_avg + 1e-6) 

    if rb_diff > 65 and bg_ratio < 0.85:
        undertone = "Warm"
    elif rb_diff < 52 or bg_ratio > 0.88: 
        undertone = "Cool"
    else:
        undertone = "Neutral"

    if val > 180:
        skin_tone = "Fair"
    elif val > 120:
        skin_tone = "Medium"
    else:
        skin_tone = "Deep"

    # Match exact properties to pick final season allocation
    avg_sat = np.mean(s_channel)
    if undertone == "Cool":
        determined_season = "Winter" if (skin_tone in ["Medium", "Deep"] or avg_sat > 40) else "Summer"
    elif undertone == "Warm":
        determined_season = "Autumn" if skin_tone in ["Medium", "Deep"] else "Spring"
    else:
        determined_season = "Summer" if skin_tone == "Fair" else "Autumn"

    # Extract clean hex from extracted skin center color point
    sampled_hex = "#%02x%02x%02x" % (int(r_avg), int(g_avg), int(b_avg))
    
    # Retrieve complete dataset corresponding to calculation models
    report_data = PALETTE_DATA[determined_season]
    
    # 3. Compile clean REST Response payload
    payload = {
        "analysis_id": str(uuid.uuid4())[:8],
        "sampled_hex": sampled_hex,
        "undertone_category": undertone,
        "skin_tone_value": skin_tone,
        **report_data
    }
    
    return Response(payload, status=status.HTTP_200_OK)