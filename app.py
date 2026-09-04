import os
import gc
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Use LiteRT / TFLite runtime instead of full TensorFlow ─────────
# TFLite = ~80MB RAM vs TensorFlow = ~450MB RAM
try:
    import ai_edge_litert.interpreter as tflite
    TFLITE_AVAILABLE = True
    print("[OK] Using Google LiteRT runtime (memory efficient)")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        print("[OK] Using TFLite runtime (memory efficient)")
    except ImportError:
        import tensorflow as tf
        TFLITE_AVAILABLE = False
        print("[WARN] Neither LiteRT nor TFLite found, falling back to TensorFlow")

app = Flask(__name__)
CORS(app, origins=['*'])

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ── Model Loading ─────────────────────────────────────────
interpreter  = None
input_index  = None
output_index = None
CLASS_NAMES  = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']
CONFIDENCE_THRESHOLD = 60.0

def load_model():
    global interpreter, input_index, output_index

    model_dir   = os.path.join(os.path.dirname(__file__), 'model')
    tflite_path = os.path.join(model_dir, 'mulberry_model.tflite')
    keras_path  = os.path.join(model_dir, 'mulberry_model.keras')

    print(f"Model dir contents: {os.listdir(model_dir)}")

    # ── Try TFLite first (preferred — low RAM) ──
    if os.path.exists(tflite_path):
        print(f"Found .tflite: {tflite_path}")
        print(f"File size: {os.path.getsize(tflite_path)/(1024*1024):.1f} MB")
        try:
            if TFLITE_AVAILABLE:
                interp = tflite.Interpreter(model_path=tflite_path)
            else:
                interp = tf.lite.Interpreter(model_path=tflite_path)

            interp.allocate_tensors()
            interpreter  = interp
            input_index  = interp.get_input_details()[0]['index']
            output_index = interp.get_output_details()[0]['index']

            print("[OK] TFLite model loaded successfully!")
            print(f"   Input:  {interp.get_input_details()[0]['shape']}")
            print(f"   Output: {interp.get_output_details()[0]['shape']}")

            # Warmup
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            interp.set_tensor(input_index, dummy)
            interp.invoke()
            print("[OK] TFLite model warmed up!")
            return
        except Exception as e:
            print(f"[ERROR] TFLite load failed: {e}")

    # ── Fallback to .keras (only if TFLite fails) ──
    if os.path.exists(keras_path):
        print(f"Falling back to .keras: {keras_path}")
        try:
            import tensorflow as tf
            global keras_model
            keras_model = tf.keras.models.load_model(keras_path, compile=False)
            print("[OK] Keras model loaded (fallback)")
            return
        except Exception as e:
            print(f"[ERROR] Keras load failed: {e}")

    print("[ERROR] No model could be loaded!")

load_model()

# ── Knowledge Bases ───────────────────────────────────────
DISEASE_KNOWLEDGE = {
    'Disease Free leaves': {
        'severity': 'None',
        'status': 'HEALTHY LEAF',
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
            'Store harvested leaves in cool damp cloth to retain freshness',
            'Use within 4–6 hours of harvest for best results'
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
        'status': 'LEAF RUST DETECTED',
        'cause': 'Fungal pathogen: Cerotelium fici. Spreads through wind-borne spores in warm, humid conditions.',
        'symptoms': [
            'Small orange-yellow pustules (uredia) on underside of leaf',
            'Corresponding yellow spots visible on upper leaf surface',
            'Premature yellowing and early leaf drop in severe cases',
            'Powdery rust-coloured spore masses on leaf underside'
        ],
        'silkworm_impact': 'DANGEROUS — Rust-affected leaves reduce silkworm appetite by 30–40%. DO NOT feed rust-infected leaves to silkworms.',
        'chemical': 'Mancozeb 75% WP (Dithane M-45) or Wettable Sulphur 80% WP',
        'dosage': 'Mancozeb: 2g per litre of water | Wettable Sulphur: 3g per litre',
        'frequency': 'Spray every 10–12 days. Stop 7 days before harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop feeding rust-affected leaves to silkworms',
            'Remove and destroy (burn) all visibly infected leaves today',
            'Spray Mancozeb 2g/L on all plants including healthy ones nearby',
            'Disinfect rearing trays with 2% bleaching powder solution'
        ],
        'prevention': [
            'Apply preventive Mancozeb spray at start of monsoon season',
            'Maintain plant spacing of at least 90cm × 90cm for airflow',
            'Plant rust-resistant mulberry varieties: S-146, MR-2, Victory-1',
            'Monitor garden weekly during humid/rainy weather'
        ]
    },
    'Leaf spot': {
        'severity': 'High',
        'status': 'LEAF SPOT DETECTED',
        'cause': 'Fungal pathogens: Pseudocercospora mori or Cercospora moricola. Thrives in warm humid environments.',
        'symptoms': [
            'Circular to irregular brown/grey spots with dark brown margins',
            'Spots range from 2–15mm in diameter on upper leaf surface',
            'Yellow halo surrounding spots in early infection stage',
            'Spots coalesce into large necrotic patches in severe cases'
        ],
        'silkworm_impact': 'CRITICAL DANGER — Leaf spot toxins directly harm silkworm digestive system. NEVER feed spotted leaves to silkworms.',
        'chemical': 'Carbendazim 50% WP (Bavistin) or Copper Oxychloride 50% WP',
        'dosage': 'Carbendazim: 1g per litre | Copper Oxychloride: 3g per litre',
        'frequency': 'Spray every 15 days during monsoon. Stop 10 days before harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop all leaf feeding from infected plants',
            'Remove and BURN infected leaves',
            'Spray Carbendazim 1g/L on all plants',
            'Disinfect entire rearing house with 2% formalin solution'
        ],
        'prevention': [
            'Apply Copper Oxychloride preventively before monsoon onset',
            'Ensure proper drainage — stagnant water worsens infection',
            'Remove and burn all infected plant debris after each season',
            'Plant disease-tolerant varieties: S-1635, G-2, MR-2'
        ]
    }
}

SILKWORM_CLIMATE = {
    'Egg / Incubation': {
        'growth_stage': 'Egg incubation period (10–12 days)',
        'temp_min': 25, 'temp_max': 26,
        'humidity_min': 80, 'humidity_max': 85,
        'too_cold': 'Delayed hatching, uneven emergence, reduced hatchability.',
        'too_hot': 'Premature hatching, egg desiccation, high mortality.',
        'too_dry': 'Egg shell hardening — larvae cannot emerge.',
        'too_humid': 'Fungal growth on egg surfaces causing mass death.'
    },
    'Instar 1': {
        'growth_stage': 'First instar (Chawki) — Days 1–4 after hatching',
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 85, 'humidity_max': 90,
        'too_cold': 'Larvae sluggish, feeding stops. Mortality 40–60% below 22°C.',
        'too_hot': 'Rapid dehydration, heat stress, body fluid loss.',
        'too_dry': 'Leaves dry fast — larvae cannot feed, starvation risk.',
        'too_humid': 'Flacherie bacterial infection risk increases dramatically.'
    },
    'Instar 2': {
        'growth_stage': 'Second instar (Chawki) — Days 5–7',
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 80, 'humidity_max': 85,
        'too_cold': 'Delayed molting, uneven growth, viral infection vulnerability.',
        'too_hot': 'Excessive moisture loss, weight gain drops 30%.',
        'too_dry': 'Leaves become crispy, larvae cluster at veins.',
        'too_humid': 'Muscardine fungal spores germinate on larval skin.'
    },
    'Instar 3': {
        'growth_stage': 'Third instar — Days 8–11',
        'temp_min': 25, 'temp_max': 26,
        'humidity_min': 75, 'humidity_max': 80,
        'too_cold': 'Silk gland development slows, rearing period extends 2–3 days.',
        'too_hot': 'Larvae pant (heat stress), reduced silk protein synthesis.',
        'too_dry': 'Reduced nutrition absorption, silk glands underdeveloped.',
        'too_humid': 'Grasserie (NPV) virus activation — body swells, skin shiny (fatal).'
    },
    'Instar 4': {
        'growth_stage': 'Fourth instar — Days 12–16',
        'temp_min': 24, 'temp_max': 26,
        'humidity_min': 70, 'humidity_max': 75,
        'too_cold': 'Silk gland development disrupted, cocoon shell thin.',
        'too_hot': 'Larvae hyperactive, cocoon weight drops 20–40%.',
        'too_dry': 'Poor conversion of leaves to silk protein.',
        'too_humid': 'Flacherie complex — chain mortality in rearing tray.'
    },
    'Instar 5': {
        'growth_stage': 'Fifth instar (Final feeding stage) — Days 17–23',
        'temp_min': 23, 'temp_max': 25,
        'humidity_min': 65, 'humidity_max': 70,
        'too_cold': 'Delayed spinning, irregular cocoons, reduced reelability.',
        'too_hot': 'Premature spinning — thin, poor-quality cocoons.',
        'too_dry': 'Loose, fluffy non-reelable cocoons (waste silk 40%).',
        'too_humid': 'Cocoons absorb moisture and mold — harvest rejected.'
    }
}

SYMPTOM_DISEASE_MAP = {
    'Swollen, shiny, translucent body'          : ['Grasserie'],
    'Larvae hanging head-down from tray'        : ['Grasserie'],
    'Milky white body fluid'                    : ['Grasserie'],
    'Skin ruptures easily, oozing fluid'        : ['Grasserie'],
    'Intersegmental swelling'                   : ['Grasserie'],
    'Body liquefies after death'                : ['Grasserie', 'Flacherie'],
    'Larvae stop feeding'                       : ['Grasserie', 'Flacherie', 'Muscardine', 'Pebrine'],
    'Soft, flaccid body'                        : ['Flacherie'],
    'Vomiting brownish gut juice'               : ['Flacherie'],
    'Watery, loose droppings'                   : ['Flacherie'],
    'Dark brown to black body color'            : ['Flacherie'],
    'Foul rotting smell'                        : ['Flacherie', 'Grasserie'],
    'Chain mortality in rearing bed'            : ['Flacherie'],
    'Body becomes stiff and hard'               : ['Muscardine'],
    'White powdery coating on body'             : ['Muscardine'],
    'Green powdery coating on body'             : ['Muscardine'],
    'Mummified chalky cadaver'                  : ['Muscardine'],
    'Dark spots on body surface'                : ['Muscardine', 'Pebrine'],
    'Uneven growth within same batch'           : ['Pebrine'],
    'Pepper-like dark spots on body'            : ['Pebrine'],
    'Undersized larvae, stunted growth'         : ['Pebrine'],
    'Irregular or failed molting'               : ['Pebrine'],
    'Thin, defective, loose cocoons'            : ['Pebrine'],
    'Deformed adult moths'                      : ['Pebrine'],
    'Reduced appetite, sluggish feeding'        : ['Pebrine', 'Flacherie', 'Muscardine'],
    'Restless movement, not settling on leaves' : ['Grasserie', 'Flacherie']
}

SILKWORM_DISEASES = {
    'Grasserie': {
        'also_known_as': 'Nuclear Polyhedrosis Virus (NPV) / Hanging disease',
        'type': 'Viral', 'severity': 'Very High', 'mortality': '80–100% if untreated',
        'description': 'Most devastating viral disease caused by BmNPV. Attacks fat body, blood cells, and silk glands.',
        'visible_signs': ['Shiny, taut integument', 'Swollen intersegmental membranes', 'Larvae hanging from tray edges', 'Milky white fluid oozing from larvae'],
        'spread': 'Contaminated mulberry leaves, contact with infected larvae, contaminated equipment.',
        'treatment': ['No cure — remove and burn all infected larvae', 'Disinfect rearing house with 2% formalin', 'Apply slaked lime powder on rearing beds', 'Stop rearing in affected room for 15 days'],
        'prevention': ['Use only certified Disease-Free Layings (DFLs)', 'Disinfect rearing house before each season', 'Wash mulberry leaves before feeding', 'Remove dead larvae twice daily']
    },
    'Flacherie': {
        'also_known_as': 'Bacterial Flacherie / Sotto disease',
        'type': 'Bacterial + Viral', 'severity': 'High', 'mortality': '30–90%',
        'description': 'Complex disease involving bacteria and Infectious Flacherie Virus. Attacks midgut.',
        'visible_signs': ['Soft, flaccid body', 'Brownish vomit around mouth', 'Dark watery frass', 'Foul rotting smell'],
        'spread': 'Contaminated leaves and frass, oral-fecal route. High humidity is major factor.',
        'treatment': ['Remove and destroy affected larvae', 'Apply lime powder on rearing beds', 'Reduce temperature to 24–25°C', 'Reduce humidity to 65–70%'],
        'prevention': ['Daily disinfection with 2% bleaching powder', 'Control temperature 24–26°C and humidity 65–75%', 'Avoid feeding wet or wilted leaves']
    },
    'Muscardine': {
        'also_known_as': 'White/Green Muscardine / Beauveria bassiana',
        'type': 'Fungal', 'severity': 'Moderate to High', 'mortality': '20–60%',
        'description': 'Caused by Beauveria bassiana. Fungal spores penetrate larval skin and mummify the body.',
        'visible_signs': ['Hard mummified cadavers with white/green powder', 'Chalky stiff larvae', 'Powder falls when touched'],
        'spread': 'Airborne fungal spores, contact with infected cadavers. High humidity favors germination.',
        'treatment': ['Remove and burn mummified cadavers', 'Reduce humidity below 70%', 'Dust beds with slaked lime + RKO', 'Sun-dry trays for 2–3 days'],
        'prevention': ['Maintain humidity 65–75%', 'Good ventilation in rearing house', 'Disinfect with 2% formalin before each crop']
    },
    'Pebrine': {
        'also_known_as': 'Nosema disease / Pepper disease',
        'type': 'Microsporidian', 'severity': 'Very High', 'mortality': '50–100%',
        'description': 'Caused by Nosema bombycis. Passes from mother moth to eggs (transovarial).',
        'visible_signs': ['Pepper-like dark spots on body', 'Uneven-sized larvae', 'Thin flimsy cocoons', 'Deformed adult moths'],
        'spread': 'Primarily transovarial. Also through spore-contaminated leaves and equipment.',
        'treatment': ['No cure — destroy entire batch', 'Disinfect with 2% formalin for 24 hours', 'Report to sericulture department'],
        'prevention': ['Use only certified DFLs from government grainage', 'Mother moth examination before using eggs', 'Disinfect before every rearing season']
    }
}

SYMPTOM_ID_TO_LABEL = {
    'body_swelling'          : 'Swollen, shiny, translucent body',
    'milky_fluid'            : 'Milky white body fluid',
    'restless_crawling'      : 'Restless movement, not settling on leaves',
    'sluggish_loss_appetite' : 'Larvae stop feeding',
    'black_rectal_protrusion': 'Soft, flaccid body',
    'foul_odor_darkening'    : 'Foul rotting smell',
    'chalky_white_mummy'     : 'White powdery coating on body',
    'loss_elasticity'        : 'Body becomes stiff and hard',
    'black_pebrine_spots'    : 'Pepper-like dark spots on body',
    'uneven_hatching_growth' : 'Uneven growth within same batch',
    'microscopic_corpuscles' : 'Deformed adult moths'
}

# ── Prediction Helper ─────────────────────────────────────
def run_inference(img_array):
    """Run inference using TFLite interpreter — low RAM usage"""
    if interpreter is not None:
        interpreter.set_tensor(input_index, img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)
    else:
        # Keras fallback
        import tensorflow as tf
        return keras_model(
            tf.constant(img_array), training=False
        ).numpy()

# ── ROUTES ────────────────────────────────────────────────
@app.route('/')
def index():
    return jsonify({
        'app': 'SeriSense AI Backend',
        'status': 'running',
        'version': '2.0.0',
        'inference': 'TFLite' if interpreter is not None else 'Keras',
        'endpoints': ['/api/health', '/api/predict-disease',
                      '/api/climate-check', '/api/diagnose-silkworm',
                      '/api/symptoms-list']
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status'              : 'running',
        'model_loaded'        : interpreter is not None,
        'inference_engine'    : 'TFLite' if interpreter is not None else 'Keras',
        'classes'             : CLASS_NAMES,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'gradcam_enabled'     : False,
        'modules'             : ['disease', 'climate', 'silkworm']
    })

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    if interpreter is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    try:
        file = request.files.get('image') or request.files.get('file')
        if not file:
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400

        # Preprocess
        file_bytes = file.read()
        img        = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
        img_array  = np.expand_dims(
            np.array(img, dtype=np.float32) / 255.0, axis=0
        )
        del file_bytes, img
        gc.collect()

        # Inference — TFLite uses minimal RAM
        raw_preds  = run_inference(img_array)
        del img_array
        gc.collect()

        scores     = {CLASS_NAMES[i]: float(raw_preds[0][i]) * 100
                      for i in range(len(CLASS_NAMES))}
        pred_class = max(scores, key=scores.get)
        confidence = scores[pred_class]
        uncertain  = confidence < CONFIDENCE_THRESHOLD

        info = DISEASE_KNOWLEDGE.get(pred_class, {}) if not uncertain else {
            'severity': 'Unknown',
            'status': 'Uncertain - please retake photo',
            'cause': 'Confidence below 60%.',
            'symptoms': ['Low confidence result'],
            'silkworm_impact': 'Retake photo in good natural light.',
            'chemical': 'N/A', 'dosage': 'N/A', 'frequency': 'N/A',
            'immediate_actions': ['Retake photo in daylight', 'Fill frame with leaf'],
            'prevention': ['Use natural daylight', 'Hold camera steady']
        }

        gc.collect()
        return jsonify({
            'success'          : True,
            'predicted_class'  : pred_class,
            'confidence'       : round(confidence, 1),
            'is_uncertain'     : uncertain,
            'gradcam_image'    : None,
            'all_scores'       : {k: round(v, 1) for k, v in scores.items()},
            'severity'         : info['severity'],
            'status'           : info['status'],
            'cause'            : info['cause'],
            'symptoms'         : info['symptoms'],
            'silkworm_impact'  : info['silkworm_impact'],
            'chemical'         : info['chemical'],
            'dosage'           : info['dosage'],
            'frequency'        : info['frequency'],
            'immediate_actions': info['immediate_actions'],
            'prevention'       : info['prevention']
        })

    except Exception as e:
        print(f"[ERROR] PREDICT: {e}")
        gc.collect()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/climate-check', methods=['POST'])
def climate_check():
    try:
        data     = request.get_json()
        stage    = data['stage']
        temp     = float(data['temperature'])
        humidity = float(data['humidity'])
        rule     = SILKWORM_CLIMATE[stage]

        issues = []; consequences = []; corrections = []
        status = 'SAFE'

        if temp < rule['temp_min']:
            diff = round(rule['temp_min'] - temp, 1)
            issues.append(f"Temperature {diff}°C BELOW ideal range")
            consequences.append(rule['too_cold'])
            corrections.append(f"Increase by {diff}°C → Target: {rule['temp_min']}–{rule['temp_max']}°C")
            status = 'WARNING' if diff <= 2 else 'CRITICAL'
        elif temp > rule['temp_max']:
            diff = round(temp - rule['temp_max'], 1)
            issues.append(f"Temperature {diff}°C ABOVE ideal range")
            consequences.append(rule['too_hot'])
            corrections.append(f"Decrease by {diff}°C → Target: {rule['temp_min']}–{rule['temp_max']}°C")
            status = 'WARNING' if diff <= 2 else 'CRITICAL'

        if humidity < rule['humidity_min']:
            diff = round(rule['humidity_min'] - humidity, 1)
            issues.append(f"Humidity {diff}% BELOW ideal range")
            consequences.append(rule['too_dry'])
            corrections.append(f"Increase by {diff}% → Use humidifier → Target: {rule['humidity_min']}–{rule['humidity_max']}% RH")
            if status != 'CRITICAL': status = 'WARNING' if diff <= 5 else 'CRITICAL'
        elif humidity > rule['humidity_max']:
            diff = round(humidity - rule['humidity_max'], 1)
            issues.append(f"Humidity {diff}% ABOVE ideal range")
            consequences.append(rule['too_humid'])
            corrections.append(f"Decrease by {diff}% → Improve ventilation → Target: {rule['humidity_min']}–{rule['humidity_max']}% RH")
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
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/diagnose-silkworm', methods=['POST'])
def diagnose_silkworm():
    try:
        data     = request.get_json()
        symptoms = data['symptoms']
        scores   = {d: 0 for d in SILKWORM_DISEASES}

        for s in symptoms:
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
            {'disease': d, 'match_percent': round(
                (sc / len(SILKWORM_DISEASES[d]['visible_signs'])) * 100, 1)}
            for d, sc in ranked[1:] if sc > 0
        ]

        return jsonify({
            'success': True, 'no_match': False,
            'top_disease'  : top_name,
            'also_known_as': info['also_known_as'],
            'type'         : info['type'],
            'severity'     : info['severity'],
            'match_percent': match_pct,
            'mortality'    : info['mortality'],
            'description'  : info['description'],
            'visible_signs': info['visible_signs'],
            'spread'       : info['spread'],
            'treatment'    : info['treatment'],
            'prevention'   : info['prevention'],
            'other_matches': others
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/symptoms-list', methods=['GET'])
def symptoms_list():
    return jsonify({'symptoms': list(SYMPTOM_DISEASE_MAP.keys())})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)))