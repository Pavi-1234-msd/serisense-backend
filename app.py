import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app, origins=['*'])

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'mulberry_model.keras')
print(f"Looking for model at: {MODEL_PATH}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
print(f"Files in model dir: {os.listdir(os.path.join(os.path.dirname(__file__), 'model'))}")
model = None
CLASS_NAMES = ['Disease Free leaves', 'Leaf Rust', 'Leaf spot']

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")

load_model()

DISEASE_KNOWLEDGE = {
    'Disease Free leaves': {
        'severity': 'NONE',
        'status': '✅ HEALTHY LEAF',
        'cause': 'No pathogen detected',
        'symptoms': [
            'Uniform bright green colour across leaf',
            'No spots, patches, or rust marks',
            'Smooth surface, no powdery coating',
            'Strong veins, no yellowing'
        ],
        'silkworm_impact': 'SAFE — Healthy leaves provide full nutritional value. Silkworms fed on healthy leaves show better cocoon weight and silk quality.',
        'chemical': 'None required',
        'dosage': 'N/A',
        'frequency': 'N/A',
        'immediate_actions': [
            'Continue feeding these leaves — they are fully safe',
            'Harvest in early morning (6–8 AM)',
            'Store in cool damp cloth to retain freshness',
            'Use within 4–6 hours of harvest'
        ],
        'prevention': [
            'Inspect leaves every 2–3 days',
            'Maintain 90cm plant spacing for airflow',
            'Apply balanced NPK fertiliser (100:50:50 kg/ha/year)',
            'Avoid overhead irrigation'
        ]
    },
    'Leaf Rust': {
        'severity': 'MODERATE',
        'status': '⚠️ LEAF RUST DETECTED',
        'cause': 'Fungal pathogen: Cerotelium fici (Obligate parasite)',
        'symptoms': [
            'Orange-yellow pustules on underside of leaf',
            'Yellow spots on upper leaf surface',
            'Premature yellowing and early leaf drop',
            'Powdery rust-coloured spore masses',
            'Leaf distortion in advanced infection'
        ],
        'silkworm_impact': 'DANGEROUS — Rust-affected leaves reduce silkworm appetite by 30–40%. Causes digestive disorders and reduces cocoon weight by up to 25%.',
        'chemical': 'Mancozeb 75% WP (Dithane M-45) or Wettable Sulphur 80% WP',
        'dosage': 'Mancozeb: 2g/L | Wettable Sulphur: 3g/L',
        'frequency': 'Every 10–12 days, minimum 3 sprays/season. Stop 7 days before harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop feeding rust-affected leaves',
            'Remove and burn all infected leaves today',
            'Spray Mancozeb 2g/L on all nearby plants',
            'Disinfect rearing trays with 2% bleaching powder',
            'Isolate infected section from healthy plants'
        ],
        'prevention': [
            'Apply preventive Mancozeb at monsoon start',
            'Maintain 90cm plant spacing',
            'Avoid excess nitrogen fertiliser',
            'Plant rust-resistant varieties: S-146, MR-2'
        ]
    },
    'Leaf spot': {
        'severity': 'HIGH',
        'status': '🔴 LEAF SPOT DETECTED',
        'cause': 'Fungal: Pseudocercospora mori or Cercospora moricola',
        'symptoms': [
            'Circular brown/grey spots with dark margins',
            'Spots 2–15mm in diameter',
            'Yellow halo surrounding spots',
            'Spots merge into large necrotic patches',
            'Premature defoliation in severe cases'
        ],
        'silkworm_impact': 'CRITICAL — Leaf spot toxins directly harm silkworm digestive system. Even 10% spotted leaves can trigger Flacherie outbreak. NEVER feed spotted leaves.',
        'chemical': 'Carbendazim 50% WP (Bavistin) or Copper Oxychloride 50% WP',
        'dosage': 'Carbendazim: 1g/L | Copper Oxychloride: 3g/L',
        'frequency': 'Every 15 days during monsoon, minimum 4 sprays. Stop 10 days before harvest.',
        'immediate_actions': [
            'IMMEDIATELY stop all feeding from infected plants',
            'Remove and BURN infected leaves',
            'Spray Carbendazim 1g/L on all plants',
            'Disinfect rearing house with 2% formalin',
            'Apply fresh lime powder on rearing house floor'
        ],
        'prevention': [
            'Apply Copper Oxychloride before monsoon',
            'Ensure proper drainage around plants',
            'Remove infected debris after each season',
            'Use drip irrigation — avoid wetting leaves'
        ]
    }
}

SILKWORM_CLIMATE = {
    'Egg / Incubation': {
        'temp_min': 24, 'temp_max': 25,
        'humidity_min': 80, 'humidity_max': 85,
        'growth_stage': 'Egg hatching stage',
        'too_hot': 'Above 26°C causes premature hatching and egg desiccation',
        'too_cold': 'Below 23°C delays hatching by 2–4 days',
        'too_humid': 'Above 90% RH promotes fungal growth on eggs',
        'too_dry': 'Below 75% RH causes egg shell hardening'
    },
    'Instar 1': {
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 85, 'humidity_max': 90,
        'growth_stage': 'First larval stage — most delicate',
        'too_hot': 'Above 29°C causes heat stress and higher mortality',
        'too_cold': 'Below 25°C severely slows growth',
        'too_humid': 'Above 92% RH promotes Flacherie infection',
        'too_dry': 'Below 80% RH causes dehydration'
    },
    'Instar 2': {
        'temp_min': 26, 'temp_max': 28,
        'humidity_min': 85, 'humidity_max': 90,
        'growth_stage': 'Second larval stage — rapid growth',
        'too_hot': 'Above 29°C triggers early moulting',
        'too_cold': 'Below 25°C reduces silk gland development',
        'too_humid': 'Above 92% increases Muscardine risk',
        'too_dry': 'Below 80% RH causes poor skin shedding'
    },
    'Instar 3': {
        'temp_min': 25, 'temp_max': 27,
        'humidity_min': 80, 'humidity_max': 85,
        'growth_stage': 'Third stage — silk gland development begins',
        'too_hot': 'Above 28°C causes energy waste',
        'too_cold': 'Below 24°C slows silk protein synthesis',
        'too_humid': 'Above 87% increases Flacherie and Pebrine risk',
        'too_dry': 'Below 75% reduces leaf intake'
    },
    'Instar 4': {
        'temp_min': 23, 'temp_max': 26,
        'humidity_min': 70, 'humidity_max': 80,
        'growth_stage': 'Fourth stage — silk protein accumulation',
        'too_hot': 'Above 28°C denatures silk proteins',
        'too_cold': 'Below 22°C causes sluggish movement',
        'too_humid': 'Above 85% promotes Grasserie viral disease',
        'too_dry': 'Below 65% causes body weight loss'
    },
    'Instar 5': {
        'temp_min': 22, 'temp_max': 25,
        'humidity_min': 65, 'humidity_max': 75,
        'growth_stage': 'Final stage — maximum silk accumulation',
        'too_hot': 'Above 26°C causes early spinning with poor cocoon',
        'too_cold': 'Below 21°C delays spinning by 1–2 days',
        'too_humid': 'Above 80% is the leading cause of Flacherie',
        'too_dry': 'Below 60% causes silk thread breakage'
    }
}

SILKWORM_DISEASES = {
    'Grasserie': {
        'also_known_as': 'Nuclear Polyhedrosis Virus (BmNPV)',
        'type': 'Viral Disease',
        'severity': 'CRITICAL',
        'symptoms': ['body_swollen','skin_shiny','body_yellowish','sluggish_movement','liquid_oozing'],
        'description': 'Most destructive viral disease. Spreads rapidly through contaminated leaves and equipment.',
        'visible_signs': [
            'Body becomes swollen and bloated',
            'Skin appears shiny and translucent',
            'Body turns yellowish-white',
            'Sluggish movement, stops eating',
            'Liquid oozes when body is pressed — highly infectious'
        ],
        'spread': 'Spreads through infected leaf, contaminated tools, and physical contact',
        'treatment': [
            'NO CURE — prevention is the only strategy',
            'Remove and burn all infected silkworms immediately',
            'Disinfect trays with 2% formalin for 30 minutes',
            'Spray 0.5% bleaching powder on rearing house floor'
        ],
        'prevention': [
            'Use only certified Disease-Free Layings (DFLs)',
            'Disinfect rearing house 48 hours before each batch',
            'Never mix different age groups of silkworms',
            'Remove dead worms daily and burn immediately'
        ],
        'mortality': 'Up to 80–100% if not controlled within 48 hours'
    },
    'Flacherie': {
        'also_known_as': 'Infectious Flacherie Virus + Bacterial',
        'type': 'Viral + Bacterial Disease',
        'severity': 'HIGH',
        'symptoms': ['soft_body','dark_patches','foul_smell','loose_droppings','less_movement'],
        'description': 'Combined viral and bacterial disease triggered by poor rearing conditions.',
        'visible_signs': [
            'Body becomes soft and flaccid',
            'Dark brownish patches on body surface',
            'Foul/sour smell from infected silkworms',
            'Loose watery droppings',
            'Reduced movement, stops climbing'
        ],
        'spread': 'Spreads through contaminated droppings and humidity above 85%',
        'treatment': [
            'Remove all infected worms and burn immediately',
            'Reduce humidity to below 75% immediately',
            'Apply RKO antibiotic (200mg/L) spray on leaves',
            'Sprinkle slaked lime powder on rearing beds'
        ],
        'prevention': [
            'Never feed diseased or wilted leaves',
            'Maintain humidity below 85%',
            'Apply 2g slaked lime per tray daily',
            'Avoid sudden temperature fluctuations'
        ],
        'mortality': '30–60% if not treated within 24 hours'
    },
    'Muscardine': {
        'also_known_as': 'White/Green Muscardine Fungus',
        'type': 'Fungal Disease',
        'severity': 'HIGH',
        'symptoms': ['white_powder_on_body','body_hard','abnormal_posture','green_powder_on_body','mummified_body'],
        'description': 'Fungal disease caused by Beauveria bassiana. Body becomes mummified after death.',
        'visible_signs': [
            'White or green powdery coating on body',
            'Body becomes rigid and hard',
            'Abnormal bent/twisted posture',
            'Mummified corpse after death',
            'Fungal growth spreading from body'
        ],
        'spread': 'Spores spread through air and contaminated equipment in cool humid conditions',
        'treatment': [
            'Remove and burn all hard/mummified silkworms',
            'Spray Thiram 75% WP (3g/L) on rearing surfaces',
            'Apply Bavistin (1g/L) on rearing trays',
            'Reduce humidity below 75%'
        ],
        'prevention': [
            'Maintain temperature above 24°C',
            'Keep humidity below 75%',
            'Apply Thiram dust preventively before each batch',
            'Use UV light periodically to kill airborne spores'
        ],
        'mortality': '20–50%, can reach 100% if spores spread'
    },
    'Pebrine': {
        'also_known_as': 'Microsporidiosis — Nosema bombycis',
        'type': 'Protozoan Disease',
        'severity': 'CRITICAL',
        'symptoms': ['pepper_spots','irregular_growth','slow_development','failure_to_moult','uneven_sizes'],
        'description': 'Most feared disease — transmitted from mother moth to eggs. Cannot be cured.',
        'visible_signs': [
            'Dark pepper-like spots on body surface',
            'Very uneven sizes in same-age silkworms',
            'Failure to moult or abnormal moulting',
            'Extremely slow development',
            'High mortality during moulting'
        ],
        'spread': 'Primary: infected mother moth to eggs. Secondary: contaminated equipment',
        'treatment': [
            'NO TREATMENT — entire batch must be destroyed',
            'Burn ALL silkworms, trays, and bedding',
            'Fumigate with formaldehyde gas',
            'Report to Sericulture Department immediately'
        ],
        'prevention': [
            'Use ONLY certified Disease-Free Layings (DFLs)',
            'Microscopic mother moth test before using eggs',
            'Destroy all moths and eggs after each cycle',
            'Maintain strict quarantine in rearing house'
        ],
        'mortality': 'Can cause 100% crop failure — entire batch must be destroyed'
    }
}

SYMPTOM_DISEASE_MAP = {
    'body_swollen': ['Grasserie'],
    'skin_shiny': ['Grasserie'],
    'body_yellowish': ['Grasserie'],
    'liquid_oozing': ['Grasserie'],
    'soft_body': ['Flacherie'],
    'dark_patches': ['Flacherie','Pebrine'],
    'foul_smell': ['Flacherie'],
    'loose_droppings': ['Flacherie'],
    'white_powder_on_body': ['Muscardine'],
    'green_powder_on_body': ['Muscardine'],
    'body_hard': ['Muscardine'],
    'mummified_body': ['Muscardine'],
    'abnormal_posture': ['Muscardine','Grasserie'],
    'pepper_spots': ['Pebrine'],
    'irregular_growth': ['Pebrine'],
    'slow_development': ['Pebrine','Flacherie'],
    'failure_to_moult': ['Pebrine'],
    'uneven_sizes': ['Pebrine'],
    'sluggish_movement': ['Grasserie','Flacherie'],
    'less_movement': ['Flacherie','Pebrine']
}

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'modules': ['disease', 'climate', 'silkworm'],
        'version': '1.0.0'
    })

@app.route('/api/predict-disease', methods=['POST'])
def predict_disease():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        confidence_scores = {
            CLASS_NAMES[i]: round(float(preds[0][i]) * 100, 1)
            for i in range(len(CLASS_NAMES))
        }
        predicted = max(confidence_scores, key=confidence_scores.get)
        info = DISEASE_KNOWLEDGE[predicted]

        return jsonify({
            'success': True,
            'predicted_class': predicted,
            'confidence': confidence_scores[predicted],
            'all_scores': confidence_scores,
            'severity': info['severity'],
            'status': info['status'],
            'cause': info['cause'],
            'symptoms': info['symptoms'],
            'silkworm_impact': info['silkworm_impact'],
            'chemical': info['chemical'],
            'dosage': info['dosage'],
            'frequency': info['frequency'],
            'immediate_actions': info['immediate_actions'],
            'prevention': info['prevention']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/climate-check', methods=['POST'])
def climate_check():
    try:
        data = request.get_json()
        stage = data.get('stage', 'Instar 1')
        temp = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 80))

        rule = SILKWORM_CLIMATE.get(stage)
        if not rule:
            return jsonify({'success': False, 'error': f'Unknown stage: {stage}'}), 400

        issues, consequences, corrections = [], [], []
        status = 'SAFE'

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
            if status != 'CRITICAL':
                status = 'WARNING' if diff <= 5 else 'CRITICAL'
        elif humidity > rule['humidity_max']:
            diff = round(humidity - rule['humidity_max'], 1)
            issues.append(f"Humidity {diff}% ABOVE ideal range")
            consequences.append(rule['too_humid'])
            corrections.append(f"Decrease humidity by {diff}% → Open ventilation → Target: {rule['humidity_min']}–{rule['humidity_max']}% RH")
            if status != 'CRITICAL':
                status = 'WARNING' if diff <= 5 else 'CRITICAL'

        return jsonify({
            'success': True,
            'stage': stage,
            'growth_stage': rule['growth_stage'],
            'current_temp': temp,
            'current_humidity': humidity,
            'ideal_temp': f"{rule['temp_min']}–{rule['temp_max']}°C",
            'ideal_humidity': f"{rule['humidity_min']}–{rule['humidity_max']}% RH",
            'status': status,
            'issues': issues,
            'consequences': consequences,
            'corrections': corrections
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/diagnose-silkworm', methods=['POST'])
def diagnose_silkworm():
    try:
        data = request.get_json()
        selected_symptoms = data.get('symptoms', [])

        disease_scores = {d: 0 for d in SILKWORM_DISEASES}
        for symptom in selected_symptoms:
            if symptom in SYMPTOM_DISEASE_MAP:
                for disease in SYMPTOM_DISEASE_MAP[symptom]:
                    disease_scores[disease] += 1

        ranked = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        top_disease, top_score = ranked[0]

        if top_score == 0:
            return jsonify({'success': True, 'no_match': True})

        info = SILKWORM_DISEASES[top_disease]
        match_pct = round((top_score / len(info['symptoms'])) * 100, 1)
        other_matches = [
            {'disease': d, 'match_percent': round((s/len(SILKWORM_DISEASES[d]['symptoms']))*100,1)}
            for d, s in ranked[1:] if s > 0
        ]

        return jsonify({
            'success': True,
            'no_match': False,
            'top_disease': top_disease,
            'also_known_as': info['also_known_as'],
            'type': info['type'],
            'severity': info['severity'],
            'match_percent': match_pct,
            'mortality': info['mortality'],
            'description': info['description'],
            'visible_signs': info['visible_signs'],
            'spread': info['spread'],
            'treatment': info['treatment'],
            'prevention': info['prevention'],
            'other_matches': other_matches
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/symptoms-list', methods=['GET'])
def symptoms_list():
    return jsonify({'symptoms': list(SYMPTOM_DISEASE_MAP.keys())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)