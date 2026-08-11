import os
import json
import gc
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)
CORS(app, origins=['*'])

# ── Model Loading ─────────────────────────────────────────
model = None
CLASS_NAMES = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']
CONFIDENCE_THRESHOLD = 60.0

def load_model():
    global model

    keras_path = os.path.join(os.path.dirname(__file__), 'model', 'mulberry_model.keras')
    h5_path    = os.path.join(os.path.dirname(__file__), 'model', 'mulberry_model.h5')

    print(f"Current directory: {os.path.dirname(__file__)}")
    model_dir  = os.path.join(os.path.dirname(__file__), 'model')
    if os.path.exists(model_dir):
        print(f"Model dir contents: {os.listdir(model_dir)}")
    else:
        print("❌ model/ directory not found!")

    if os.path.exists(keras_path):
        print(f"Found .keras file: {keras_path}")
        print(f"File size: {os.path.getsize(keras_path) / (1024*1024):.1f} MB")
        try:
            model = tf.keras.models.load_model(keras_path, compile=False)
            print(f"✅ Model loaded from .keras!")
            print(f"   Input shape:  {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return
        except Exception as e:
            print(f"❌ .keras load failed: {e}")

    if os.path.exists(h5_path):
        print(f"Found .h5 file: {h5_path}")
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
            print(f"✅ Model loaded from .h5!")
            print(f"   Input shape:  {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            return
        except Exception as e:
            print(f"❌ .h5 load failed: {e}")

    print("❌ No model file found or all load attempts failed")

load_model()

# ── Grad-CAM helpers ──────────────────────────────────────
def get_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def make_gradcam(img_norm, pred_index):
    last_conv = get_last_conv_layer(model)
    if last_conv is None:
        return None
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_norm)
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        res = heatmap.numpy()
        del grad_model, tape, grads, pooled, conv_out
        return res
    except Exception as e:
        print(f"Grad-CAM skipped (memory save): {e}")
        return None

def overlay_gradcam(img_uint8, heatmap, alpha=0.4):
    img     = cv2.resize(img_uint8.astype('uint8'), (224, 224))
    hm      = cv2.resize(heatmap, (224, 224))
    hm_uint = np.uint8(255 * hm)
    hm_col  = cv2.applyColorMap(hm_uint, cv2.COLORMAP_JET)
    hm_col  = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, hm_col, alpha, 0)

def img_to_b64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb.astype('uint8'), cv2.COLOR_RGB2BGR)
    _, buf  = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')

# ── Knowledge Bases ───────────────────────────────────────
DISEASE_KNOWLEDGE = {
    'Disease Free leaves': {
        'severity': 'None',
        'status': '✅ HEALTHY LEAF',
        'cause': 'No pathogen detected.',
        'symptoms': [
            'Uniform bright green colour across entire leaf',
            'No spots, patches, rust marks or discolouration',
            'Smooth leaf surface with no powdery coating',
            'Strong leaf veins with no yellowing at margins'
        ],
        'silkworm_impact': 'SAFE — Healthy leaves provide full nutritional value. Silkworms fed on healthy leaves show better growth, higher cocoon weight, and stronger silk thread quality.',
        'chemical': 'None required',
        'dosage': 'N/A',
        'frequency': 'N/A',
        'immediate_actions': [
            'Continue feeding these leaves to silkworms — fully safe',
            'Harvest in early morning (6–8 AM) when leaf moisture is optimal',
            'Store harvested leaves in cool, damp cloth to retain freshness',
            'Use within 4–6 hours of harvest for best silkworm feeding results'
        ],
        'prevention': [
            'Inspect leaves every 2–3 days to catch early disease signs',
            'Maintain proper plant spacing (90cm × 90cm) for airflow',
            'Apply balanced NPK fertiliser (100:50:50 kg/hectare/year)',
            'Avoid overhead irrigation — water at base of plants only'
        ]
    },
    'Leaf Rust': {
        'severity': 'Moderate',
        'status': '⚠️ LEAF RUST DETECTED',
        'cause': 'Fungal pathogen: Cerotelium fici (Obligate parasite). Spreads through wind-borne spores in warm, humid conditions.',
        'symptoms': [
            'Small orange-yellow pustules (uredia) on underside of leaf',
            'Corresponding yellow spots visible on upper leaf surface',
            'Premature yellowing and early leaf drop in severe cases',
            'Powdery rust-coloured spore masses on leaf underside'
        ],
        'silkworm_impact': 'DANGEROUS — Rust-affected leaves reduce silkworm appetite by 30–40%. Severely infected leaves cause digestive disorders and reduce cocoon weight by up to 25%. DO NOT feed rust-infected leaves to silkworms.',
        'chemical': 'Mancozeb 75% WP (Dithane M-45) or Wettable Sulphur 80% WP',
        'dosage': 'Mancozeb: 2g per litre of water | Wettable Sulphur: 3g per litre',
        'frequency': 'Spray every 10–12 days. Minimum 3 sprays per season. Stop spraying 7 days before leaf harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop feeding rust-affected leaves to silkworms',
            'Remove and destroy (burn) all visibly infected leaves today',
            'Do NOT compost infected leaves — spores will survive and spread',
            'Spray Mancozeb 2g/L on all plants including healthy ones nearby',
            'Disinfect rearing trays and tools with 2% bleaching powder solution'
        ],
        'prevention': [
            'Apply preventive Mancozeb spray at start of monsoon season',
            'Maintain plant spacing of at least 90cm × 90cm for airflow',
            'Avoid excess nitrogen fertiliser',
            'Plant rust-resistant mulberry varieties: S-146, MR-2, Victory-1',
            'Monitor garden weekly especially during humid/rainy weather'
        ]
    },
    'Leaf spot': {
        'severity': 'High',
        'status': '🔴 LEAF SPOT DETECTED',
        'cause': 'Fungal pathogens: Pseudocercospora mori or Cercospora moricola. Thrives in warm humid environments (20–28°C, >85% RH).',
        'symptoms': [
            'Circular to irregular brown/grey spots with dark brown margins',
            'Spots range from 2–15mm in diameter on upper leaf surface',
            'Yellow halo surrounding spots in early infection stage',
            'Spots coalesce into large necrotic patches in severe cases',
            'Premature defoliation (leaf drop) weakening the mulberry plant'
        ],
        'silkworm_impact': 'CRITICAL DANGER — Leaf spot toxins directly harm silkworm digestive system. Feeding spotted leaves causes Flacherie in silkworms, leading to mass mortality. Even 10% spotted leaves mixed with healthy leaves can trigger an outbreak. NEVER feed spotted leaves.',
        'chemical': 'Carbendazim 50% WP (Bavistin) or Copper Oxychloride 50% WP',
        'dosage': 'Carbendazim: 1g per litre | Copper Oxychloride: 3g per litre',
        'frequency': 'Spray every 15 days during monsoon. Minimum 4 sprays per season. Stop 10 days before harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop all leaf feeding from infected plants',
            'Remove and BURN infected leaves — do not leave on ground',
            'Spray Carbendazim 1g/L on all plants including surrounding healthy ones',
            'Check silkworms for Flacherie symptoms if spotted leaves were already fed',
            'Disinfect entire rearing house with 2% formalin solution'
        ],
        'prevention': [
            'Apply Copper Oxychloride preventively before monsoon onset',
            'Ensure proper drainage — stagnant water worsens infection',
            'Remove and burn all infected plant debris after each season',
            'Avoid wetting leaves during irrigation — drip irrigation preferred',
            'Plant disease-tolerant varieties: S-1635, G-2, MR-2'
        ]
    }
}

SILKWORM_CLIMATE = {
    'Egg / Incubation': {
        'growth_stage': 'Egg incubation period (10–12 days)',
        'temp_min': 25, 'temp_max': 26,
        'humidity_min': 80, 'humidity_max': 85,
        'too_cold': 'Delayed hatching, uneven emergence, reduced hatchability, weak first-instar larvae.',
        'too_hot': 'Premature/abnormal hatching, egg desiccation, high mortality within 24 hours.',
        'too_dry': 'Egg shell hardening and membrane desiccation — larvae cannot emerge.',
        'too_humid': 'Fungal growth (Aspergillus) on egg surfaces causing mass death.'
    },
    'Instar 1': {
        'growth_stage': 'First instar (Chawki) — Days 1–4 after hatching',
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 85, 'humidity_max': 90,
        'too_cold': 'Larvae become sluggish, feeding stops. Mortality can reach 40–60% below 22°C.',
        'too_hot': 'Rapid dehydration, heat stress, body fluid loss.',
        'too_dry': 'Chopped leaves dry out fast — larvae cannot feed, starvation risk.',
        'too_humid': 'Flacherie bacterial infection risk increases dramatically.'
    },
    'Instar 2': {
        'growth_stage': 'Second instar (Chawki) — Days 5–7',
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 80, 'humidity_max': 85,
        'too_cold': 'Delayed molting, uneven growth, prolonged viral infection vulnerability.',
        'too_hot': 'Excessive moisture loss, restless movement, weight gain drops 30%.',
        'too_dry': 'Leaves become crispy, larvae cluster at leaf veins seeking moisture.',
        'too_humid': 'Muscardine fungal spores germinate on larval skin within 48 hours.'
    },
    'Instar 3': {
        'growth_stage': 'Third instar — Days 8–11',
        'temp_min': 25, 'temp_max': 26,
        'humidity_min': 75, 'humidity_max': 80,
        'too_cold': 'Silk gland development slows, rearing period extends by 2–3 days.',
        'too_hot': 'Larvae pant and lift heads (heat stress), reduced silk protein synthesis.',
        'too_dry': 'Reduced nutrition absorption, silk glands remain underdeveloped.',
        'too_humid': 'Grasserie (NPV) virus activation — body swells, skin turns shiny (fatal).'
    },
    'Instar 4': {
        'growth_stage': 'Fourth instar — Days 12–16',
        'temp_min': 24, 'temp_max': 26,
        'humidity_min': 70, 'humidity_max': 75,
        'too_cold': 'Major silk gland development disrupted, cocoon shell will be thin and unviable.',
        'too_hot': 'Larvae hyperactive, waste energy, cocoon weight drops 20–40%.',
        'too_dry': 'Larvae consume 30% more leaves but convert poorly to silk protein.',
        'too_humid': 'Flacherie complex — larvae vomit gut juice, chain mortality in rearing tray.'
    },
    'Instar 5': {
        'growth_stage': 'Fifth instar (Final feeding stage) — Days 17–23',
        'temp_min': 23, 'temp_max': 25,
        'humidity_min': 65, 'humidity_max': 70,
        'too_cold': 'Delayed spinning, irregular/double cocoons, reduced reelability.',
        'too_hot': 'Premature spinning with insufficient silk — thin, poor-quality cocoons.',
        'too_dry': 'Loose, fluffy non-reelable cocoons (waste silk percentage increases to 40%).',
        'too_humid': 'Cocoons absorb moisture and mold — entire harvest rejected at market.'
    }
}

SYMPTOM_DISEASE_MAP = {
    'Swollen, shiny, translucent body'           : ['Grasserie'],
    'Larvae hanging head-down from tray'         : ['Grasserie'],
    'Milky white body fluid'                     : ['Grasserie'],
    'Skin ruptures easily, oozing fluid'         : ['Grasserie'],
    'Intersegmental swelling'                    : ['Grasserie'],
    'Body liquefies after death'                 : ['Grasserie', 'Flacherie'],
    'Larvae stop feeding'                        : ['Grasserie', 'Flacherie', 'Muscardine', 'Pebrine'],
    'Soft, flaccid body'                         : ['Flacherie'],
    'Vomiting brownish gut juice'                : ['Flacherie'],
    'Watery, loose droppings'                    : ['Flacherie'],
    'Dark brown to black body color'             : ['Flacherie'],
    'Foul rotting smell'                         : ['Flacherie', 'Grasserie'],
    'Chain mortality in rearing bed'             : ['Flacherie'],
    'Body becomes stiff and hard'                : ['Muscardine'],
    'White powdery coating on body'              : ['Muscardine'],
    'Green powdery coating on body'              : ['Muscardine'],
    'Mummified chalky cadaver'                   : ['Muscardine'],
    'Dark spots on body surface'                 : ['Muscardine', 'Pebrine'],
    'Uneven growth within same batch'            : ['Pebrine'],
    'Pepper-like dark spots on body'             : ['Pebrine'],
    'Undersized larvae, stunted growth'          : ['Pebrine'],
    'Irregular or failed molting'                : ['Pebrine'],
    'Thin, defective, loose cocoons'             : ['Pebrine'],
    'Deformed adult moths'                       : ['Pebrine'],
    'Reduced appetite, sluggish feeding'         : ['Pebrine', 'Flacherie', 'Muscardine'],
    'Restless movement, not settling on leaves'  : ['Grasserie', 'Flacherie']
}

SILKWORM_DISEASES = {
    'Grasserie': {
        'also_known_as': 'Nuclear Polyhedrosis Virus (NPV) / Hanging disease',
        'type': 'Viral', 'severity': 'Very High', 'mortality': '80–100% if untreated',
        'description': 'Most devastating viral disease caused by Bombyx mori Nuclear Polyhedrosis Virus (BmNPV). Attacks fat body, blood cells, and silk glands.',
        'visible_signs': ['Shiny, taut integument (oily look)', 'Swollen intersegmental membranes', 'Larvae hanging from tray edges', 'Milky white fluid oozing from ruptured larvae'],
        'spread': 'Contaminated mulberry leaves, contact with infected larvae, contaminated equipment, adult moth transmission.',
        'treatment': ['No cure — remove and burn all infected larvae immediately', 'Disinfect rearing house with 2% formalin', 'Apply slaked lime powder on rearing beds', 'Stop rearing in affected room for 15 days', 'Wash trays with 5% bleaching powder and sun-dry for 3 days'],
        'prevention': ['Use only certified Disease-Free Layings (DFLs)', 'Disinfect rearing house before each rearing season', 'Wash mulberry leaves before feeding', 'Remove dead larvae twice daily', 'Use bed disinfectants (Vijetha/RKO) after every molt']
    },
    'Flacherie': {
        'also_known_as': 'Bacterial Flacherie / Sotto disease',
        'type': 'Bacterial + Viral', 'severity': 'High', 'mortality': '30–90%',
        'description': 'Complex disease involving Bacillus thuringiensis and Infectious Flacherie Virus. Attacks midgut, disrupting digestion. Triggered by poor rearing conditions.',
        'visible_signs': ['Soft, flaccid body', 'Brownish vomit around mouth', 'Dark watery frass (not pellet-shaped)', 'Foul rotting smell in rearing room'],
        'spread': 'Contaminated leaves and frass, oral-fecal route, contaminated trays. High humidity and temperature are major predisposing factors.',
        'treatment': ['Remove and destroy affected larvae by burning', 'Apply lime powder liberally on rearing beds', 'Spray Labex (0.3%) on remaining healthy larvae', 'Reduce temperature to 24–25°C immediately', 'Reduce humidity to 65–70% by improving ventilation'],
        'prevention': ['Daily disinfection of floor with 2% bleaching powder', 'Control temperature at 24–26°C and humidity at 65–75%', 'Avoid feeding wet or wilted leaves', 'Apply bed disinfectant after every molt']
    },
    'Muscardine': {
        'also_known_as': 'White/Green Muscardine / Beauveria bassiana infection',
        'type': 'Fungal', 'severity': 'Moderate to High', 'mortality': '20–60%',
        'description': 'Caused by Beauveria bassiana (White) or Metarhizium anisopliae (Green). Fungal spores penetrate larval skin, grow internally, and mummify the body.',
        'visible_signs': ['Hard, mummified cadavers covered in white/green powder', 'Chalky, stiff larvae that do not decompose', 'Powder falls from dead larvae when touched'],
        'spread': 'Airborne fungal spores, contact with infected cadavers, contaminated trays. High humidity (>80%) strongly favors spore germination.',
        'treatment': ['Remove and burn all mummified cadavers immediately', 'Reduce rearing room humidity below 70%', 'Dust beds with slaked lime + RKO powder', 'Spray 0.3% Dithane M-45 on room walls and equipment', 'Sun-dry all trays for 2–3 consecutive days'],
        'prevention': ['Maintain humidity at 65–75% — avoid damp conditions', 'Ensure good ventilation in rearing house', 'Disinfect with 2% formalin before each crop', 'Use bed disinfectant after every molt']
    },
    'Pebrine': {
        'also_known_as': 'Nosema disease / Pepper disease',
        'type': 'Microsporidian (Protozoan)', 'severity': 'Very High (Chronic)', 'mortality': '50–100% across generations',
        'description': 'Caused by Nosema bombycis. Infects every tissue including silk glands and reproductive organs. Passes from mother moth to eggs (transovarial transmission).',
        'visible_signs': ['Pepper-like dark spots on larval body', 'Uneven-sized larvae in same tray (some much smaller)', 'Thin, flimsy cocoons easily crushed by hand', 'Deformed, weak adult moths with crumpled wings'],
        'spread': 'Primarily transovarial (infected mother → eggs). Also through spore-contaminated leaves, infected frass, and contaminated equipment.',
        'treatment': ['No cure — destroy entire infected batch by burning', 'All eggs from infected batch must be destroyed', 'Disinfect rearing house with 2% formalin for 24 hours', 'Report to nearest sericulture department immediately'],
        'prevention': ['MOST CRITICAL: Use only certified DFLs from government grainage', 'Mother moth microscopic examination before using eggs', 'Disinfect rearing house before every rearing season', 'Do not use eggs from unknown or unexamined sources']
    }
}

# ── Frontend Symptom ID → Backend Label Mapping ──────────
# The React frontend sends short IDs (e.g. 'body_swelling'),
# but SYMPTOM_DISEASE_MAP uses full descriptive labels as keys.
# This map bridges that gap.
SYMPTOM_ID_TO_LABEL = {
    'body_swelling':          'Swollen, shiny, translucent body',
    'milky_fluid':            'Milky white body fluid',
    'restless_crawling':      'Restless movement, not settling on leaves',
    'sluggish_loss_appetite': 'Larvae stop feeding',
    'black_rectal_protrusion':'Soft, flaccid body',
    'foul_odor_darkening':    'Foul rotting smell',
    'chalky_white_mummy':     'White powdery coating on body',
    'loss_elasticity':        'Body becomes stiff and hard',
    'black_pebrine_spots':    'Pepper-like dark spots on body',
    'uneven_hatching_growth': 'Uneven growth within same batch',
    'microscopic_corpuscles': 'Deformed adult moths'
}

# ── ROUTES ────────────────────────────────────────────────

@app.route('/')
def index():
    return jsonify({
        'app': 'SeriSense AI Backend',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': ['/api/health', '/api/predict-disease', '/api/climate-check', '/api/diagnose-silkworm', '/api/symptoms-list']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'classes': CLASS_NAMES,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'gradcam_enabled': True,
        'modules': ['disease', 'climate', 'silkworm']
    })

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    try:
        file = request.files.get('image') or request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        img         = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_raw     = np.array(img)
        img_norm    = np.expand_dims(img_raw / 255.0, axis=0)

        # Ultra-fast forward pass (< 50ms) using direct tensor call instead of model.predict (avoids Gunicorn thread deadlock & OOM)
        raw_preds   = model(img_norm, training=False).numpy()
        scores      = {CLASS_NAMES[i]: float(raw_preds[0][i]) * 100 for i in range(len(CLASS_NAMES))}
        pred_class  = max(scores, key=scores.get)
        confidence  = scores[pred_class]
        uncertain   = confidence < CONFIDENCE_THRESHOLD

        # Grad-CAM disabled on free tier to prevent 512MB RAM SIGKILL
        gradcam_b64 = None

        if uncertain:
            info = {
                'severity': 'Unknown', 'status': '⚠️ Uncertain — please retake photo',
                'cause': 'Confidence below 60%. Try better lighting, steady camera, single leaf in frame.',
                'symptoms': ['Confidence below safe threshold'],
                'silkworm_impact': 'Do not act on this result. Retake photo in good natural light.',
                'chemical': 'N/A', 'dosage': 'N/A', 'frequency': 'N/A',
                'immediate_actions': ['Retake photo in natural daylight', 'Fill frame with a single leaf', 'Avoid shadows and blur'],
                'prevention': ['Use daylight, not artificial light', 'Hold camera steady']
            }
        else:
            info = DISEASE_KNOWLEDGE[pred_class]

        response_data = jsonify({
            'success': True,
            'predicted_class': pred_class,
            'confidence': round(confidence, 1),
            'is_uncertain': uncertain,
            'gradcam_image': gradcam_b64,
            'all_scores': {k: round(v, 1) for k, v in scores.items()},
            **{k: info[k] for k in ['severity','status','cause','symptoms','silkworm_impact','chemical','dosage','frequency','immediate_actions','prevention']}
        })
        del img, img_raw, img_norm, raw_preds
        gc.collect()
        return response_data
    except Exception as e:
        print(f"❌ PREDICT ERROR: {e}")
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/climate-check', methods=['POST'])
def climate_check():
    try:
        data     = request.get_json()
        stage    = data['stage']
        temp     = float(data['temperature'])
        humidity = float(data['humidity'])

        rule        = SILKWORM_CLIMATE[stage]
        issues      = []
        consequences= []
        corrections = []
        status      = 'SAFE'

        if temp < rule['temp_min']:
            diff = round(rule['temp_min'] - temp, 1)
            issues.append(f"Temperature {diff}°C BELOW ideal range")
            consequences.append(rule['too_cold'])
            corrections.append(f"Increase temperature by {diff}°C → Target: {rule['temp_min']}–{rule['temp_max']}°C")
            status = 'WARNING' if diff <= 2 else 'CRITICAL'
        elif temp > rule['temp_max']:
            diff = round(temp - rule['temp_max'], 1)
            issues.append(f"Temperature {diff}°C ABOVE ideal range")
            consequences.append(rule['too_hot'])
            corrections.append(f"Decrease temperature by {diff}°C → Target: {rule['temp_min']}–{rule['temp_max']}°C")
            status = 'WARNING' if diff <= 2 else 'CRITICAL'

        if humidity < rule['humidity_min']:
            diff = round(rule['humidity_min'] - humidity, 1)
            issues.append(f"Humidity {diff}% BELOW ideal range")
            consequences.append(rule['too_dry'])
            corrections.append(f"Increase humidity by {diff}% → Use humidifier → Target: {rule['humidity_min']}–{rule['humidity_max']}% RH")
            if status != 'CRITICAL': status = 'WARNING' if diff <= 5 else 'CRITICAL'
        elif humidity > rule['humidity_max']:
            diff = round(humidity - rule['humidity_max'], 1)
            issues.append(f"Humidity {diff}% ABOVE ideal range")
            consequences.append(rule['too_humid'])
            corrections.append(f"Decrease humidity by {diff}% → Improve ventilation → Target: {rule['humidity_min']}–{rule['humidity_max']}% RH")
            if status != 'CRITICAL': status = 'WARNING' if diff <= 5 else 'CRITICAL'

        return jsonify({
            'success': True, 'stage': stage,
            'growth_stage': rule['growth_stage'],
            'current_temp': temp, 'current_humidity': humidity,
            'ideal_temp': f"{rule['temp_min']}–{rule['temp_max']}°C",
            'ideal_humidity': f"{rule['humidity_min']}–{rule['humidity_max']}% RH",
            'status': status, 'issues': issues,
            'consequences': consequences, 'corrections': corrections
        })
    except Exception as e:
        print(f"❌ CLIMATE ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/diagnose-silkworm', methods=['POST'])
def diagnose_silkworm():
    try:
        data     = request.get_json()
        symptoms = data['symptoms']

        scores = {d: 0 for d in SILKWORM_DISEASES}
        for s in symptoms:
            # Try direct label match first, then reverse-lookup from frontend ID
            label = s if s in SYMPTOM_DISEASE_MAP else SYMPTOM_ID_TO_LABEL.get(s, s)
            if label in SYMPTOM_DISEASE_MAP:
                for d in SYMPTOM_DISEASE_MAP[label]:
                    if d in scores:
                        scores[d] += 1

        ranked    = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_name  = ranked[0][0]
        top_score = ranked[0][1]

        if top_score == 0:
            return jsonify({'success': True, 'no_match': True})

        info      = SILKWORM_DISEASES[top_name]
        match_pct = round((top_score / len(info['visible_signs'])) * 100, 1)
        others    = [
            {'disease': d, 'match_percent': round((sc / len(SILKWORM_DISEASES[d]['visible_signs'])) * 100, 1)}
            for d, sc in ranked[1:] if sc > 0
        ]

        return jsonify({
            'success': True, 'no_match': False,
            'top_disease': top_name,
            'also_known_as': info['also_known_as'],
            'type': info['type'], 'severity': info['severity'],
            'match_percent': match_pct, 'mortality': info['mortality'],
            'description': info['description'],
            'visible_signs': info['visible_signs'],
            'spread': info['spread'],
            'treatment': info['treatment'],
            'prevention': info['prevention'],
            'other_matches': others
        })
    except Exception as e:
        print(f"❌ DIAGNOSE ERROR: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/symptoms-list', methods=['GET'])
def symptoms_list():
    return jsonify({'symptoms': list(SYMPTOM_DISEASE_MAP.keys())})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))