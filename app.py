import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
model = load_model("hybrid_cnn_svm_model.h5")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/inner-page')
def inner_page():
    return render_template('inner-page.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            if 'image' not in request.files:
                return "No file part", 400

            f = request.files['image']
            if f.filename == '':
                return "No selected file", 400

            basepath = os.path.dirname(__file__)
            uploads_dir = os.path.join(basepath, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)

            filepath = os.path.join(uploads_dir, f.filename)
            f.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            preds = model.predict(x)

            index = [
                "Alpinia Galanga (Rasna)", "Amaranthus Viridis (Arive-Dantu)",
                "Artocarpus Heterophyllus (Jackfruit)", "Azadirachta Indica (Neem)",
                "Basella Alba (Basale)", "Brassica Juncea (Indian Mustard)",
                "Carissa Carandas (Karanda)", "Citrus Limon (Lemon)",
                "Ficus Auriculata (Roxburgh fig)", "Ficus Religiosa (Peepal Tree)",
                "Hibiscus Rosa-sinensis", "Jasminum (Jasmine)",
                "Mangifera Indica (Mango)", "Mentha (Mint)",
                "Moringa Oleifera (Drumstick)", "Muntingia Calabura (Jamaica Cherry-Gasagase)",
                "Murraya Koenigii (Curry)", "Nerium Oleander (Oleander)",
                "Nyctanthes Arbor-tristis (Parijata)", "Ocimum Tenuiflorum (Tulsi)",
                "Piper Betle (Betel)", "Plectranthus Amboinicus (Mexican Mint)",
                "Pongamia Pinnata (Indian Beech)", "Psidium Guajava (Guava)",
                "Punica Granatum (Pomegranate)", "Santalum Album (Sandalwood)",
                "Syzygium Cumini (Jamun)", "Syzygium Jambos (Rose Apple)",
                "Tabernaemontana Divaricata (Crape Jasmine)",
                "Trigonella Foenum-graecum (Fenugreek)"
            ]

            medicinal_plants = {
                "Alpinia Galanga (Rasna)", "Azadirachta Indica (Neem)",
                "Basella Alba (Basale)", "Brassica Juncea (Indian Mustard)",
                "Carissa Carandas (Karanda)", "Citrus Limon (Lemon)",
                "Ficus Auriculata (Roxburgh fig)", "Ficus Religiosa (Peepal Tree)",
                "Hibiscus Rosa-sinensis", "Jasminum (Jasmine)",
                "Mangifera Indica (Mango)", "Mentha (Mint)",
                "Moringa Oleifera (Drumstick)", "Murraya Koenigii (Curry)",
                "Nerium Oleander (Oleander)", "Nyctanthes Arbor-tristis (Parijata)",
                "Ocimum Tenuiflorum (Tulsi)", "Piper Betle (Betel)",
                "Plectranthus Amboinicus (Mexican Mint)", "Psidium Guajava (Guava)",
                "Punica Granatum (Pomegranate)", "Santalum Album (Sandalwood)",
                "Syzygium Cumini (Jamun)", "Tabernaemontana Divaricata (Crape Jasmine)",
                "Trigonella Foenum-graecum (Fenugreek)"
            }

            # Dictionary for storing additional plant information
            plant_info = {
                "Alpinia Galanga (Rasna)": {
                    "common_name": "Rasna",
                    "scientific_name": "Alpinia Galanga",
                    "part_used": "Leaves, Root",
                    "cures":"Inflammation, Joint Pain",
                    "uses": "Used in Ayurvedic medicine for digestive disorders and inflammation."
                },
                "Amaranthus Viridis (Arive-Dantu)": {
                    "common_name": "Arive-Dantu",
                    "scientific_name": "Amaranthus Viridis",
                    "part_used": "Whole Plant",
                    "cures":"Kidney Stone, Urinary Disorders",
                    "uses": "Leaves are used as a vegetable; seeds are consumed for their nutritional value."
                },
                "Artocarpus Heterophyllus (Jackfruit)": {
                    "common_name": "Jackfruit",
                    "scientific_name": "Artocarpus Heterophyllus",
                    "part_used": "Fruits, Seeds, Leaves",
                    "cures":"Wounds, Digestive Issues",
                    "uses": "Fruit is consumed as food; seeds have been used traditionally for their nutritional content."
                },
                "Azadirachta Indica (Neem)": {
                    "common_name": "Neem",
                    "scientific_name": "Azadirachta Indica",
                    "part_used": "Leaves, Bark, Seeds",
                    "cures":"Skin Disorder, Fever, Dental Issues",
                    "uses": "Used for skin disorders, antibacterial properties, and as an immune booster."
                },
                "Basella Alba (Basale)": {
                    "common_name": "Basale",
                    "scientific_name": "Basella Alba",
                    "part_used": "Leaves, Stems",
                    "cures":"Inflammation",
                    "uses": "Leaves used as a laxative; beneficial in treating diarrhea and anemia."
                },
                "Brassica Juncea (Indian Mustard)": {
                    "common_name": "Indian Mustard",
                    "scientific_name": "Brassica Juncea",
                    "part_used": "Leaves, Seeds, Oil",
                    "cures":"Joint Pain, Cough",
                    "uses": "Seeds used for oil extraction; leaves consumed as a vegetable."
                },
                "Carissa Carandas (Karanda)": {
                    "common_name": "Karanda",
                    "scientific_name": "Carissa Carandas",
                    "part_used": "Fruits",
                    "cures":"Anemia, Digestive Issues",
                    "uses": "Fruit used in pickles and jams; plant has applications in traditional medicine."
                },
                "Citrus Limon (Lemon)": {
                    "common_name": "Lemon",
                    "scientific_name": "Citrus Limon",
                    "part_used": "Fruits, Juice, Peel",
                    "cures":"Sore Throat, Indigestion",
                    "uses": "Rich in vitamin C; used in culinary applications and traditional remedies."
                },
                "Ficus Auriculata (Roxburgh fig)": {
                    "common_name": "Roxburgh fig",
                    "scientific_name": "Ficus Auriculata",
                    "uses": "Fruits are edible; leaves used as fodder."
                },
                "Ficus Religiosa (Peepal Tree)": {
                    "common_name": "Peepal Tree",
                    "scientific_name": "Ficus Religiosa",
                    "part_used": "Leaves,Bark, Root",
                    "cures":"Asthma,Diabetes,Skin Diseases",
                    "uses": "Sacred tree in India; leaves and bark used in traditional medicine."
                },
                "Hibiscus Rosa-sinensis": {
                    "common_name": "Hibiscus",
                    "scientific_name": "Hibiscus Rosa-sinensis",
                    "part_used": "Leaves, Flowers",
                    "cures":"Hairfall, High Blood Pressure",
                    "uses": "Flowers used in hair care products; has potential medicinal applications."
                },
                "Jasminum (Jasmine)": {
                    "common_name": "Jasmine",
                    "scientific_name": "Jasminum",
                    "part_used": "Leaves, Flowers",
                    "cures":"Stress, Headache",
                    "uses": "Flowers used for their fragrance; has applications in perfumery."
                },
                "Mangifera Indica (Mango)": {
                    "common_name": "Mango",
                    "scientific_name": "Mangifera Indica",
                    "part_used": "Leaves, Leaves, Bark",
                    "cures":"Diarrhea, Diabetes, Respiratory Issues",
                    "uses": "Fruit consumed worldwide; leaves and bark used in traditional medicine."
                },
                "Ocimum Tenuiflorum (Tulsi)": {
                    "common_name": "Holy Basil",
                    "scientific_name": "Ocimum Tenuiflorum",
                    "part_used": "Leaves",
                    "cures": "Cold,Cough",
                    "uses": "Used for respiratory disorders, stress relief, and boosting immunity."
                },
                "Mentha (Mint)": {
                    "common_name": "Mint",
                    "scientific_name": "Mentha",
                    "part_used": "Leaves",
                    "cures":"Indigestion, Bad breath",
                    "uses": "Used for digestion, nausea relief, and cooling effects."
                },
                "Moringa Oleifera (Drumstick)": {
                    "common_name": "Drumstick",
                    "scientific_name": "Moringa Oleifera",
                    "part_used": "Leaves, Pods, Seeds",
                    "cures":"Inflammation, Anemia",
                    "uses": "Rich in vitamins and used to reduce inflammation."
                },
                "Muntingia Calabura (Jamaica Cherry-Gasagase)": {
                    "common_name": "Jamaica Cherry",
                    "scientific_name": "Muntingia Calabura",
                    "part_used": "Fruits, Leaves",
                    "cures": "None (Non-medicinal)",
                    "uses": "Fruits are edible; used in traditional medicine in some cultures."
                },
                "Murraya Koenigii (Curry)": {
                    "common_name": "Curry",
                    "scientific_name": "Murraya Koenigii",
                    "part_used": "Leaves",
                    "cures": "Diarrhea, Skin Infections",
                    "uses": "Leaves used in cooking; believed to have medicinal properties."
                },
                "Nerium Oleander (Oleander)": {
                    "common_name": "Oleander",
                    "scientific_name": "Nerium Oleander",
                    "part_used": "None",
                    "cures":"None (Poisonous)",
                    "uses": "Ornamental plant; note that all parts are toxic if ingested."
                },
                "Nyctanthes Arbor-tristis (Parijata)": {
                    "common_name": "Parijata",
                    "scientific_name": "Nyctanthes Arbor-tristis",
                    "part_used": "Leaves, Flowers, Bark",
                    "cures":"Fever, Joint Pain",
                    "uses": "Flowers used in traditional medicine; leaves have medicinal applications."
                },
                "Piper Betle (Betel)": {
                    "common_name": "Betel",
                    "scientific_name": "Piper Betle",
                    "uses": "Leaves chewed with areca nut; has cultural significance."
                },
                "Plectranthus Amboinicus (Mexican Mint)": {
                    "common_name": "Mexican Mint",
                    "scientific_name": "Plectranthus Amboinicus",
                    "uses": "Leaves used in traditional medicine; aromatic properties."
                },
                "Pongamia Pinnata (Indian Beech)": {
                    "common_name": "Indian Beech",
                    "scientific_name": "Pongamia Pinnata",
                    "part_used": "Leaves, Seeds, Bark",
                    "cures":"Skin Diseases",
                    "uses": "Seeds used for oil extraction; tree has applications in agroforestry."
                },
                "Psidium Guajava (Guava)": {
                    "common_name": "Guava",
                    "scientific_name": "Psidium Guajava",
                    "part_used": "Leaves, Fruits",
                    "cures":"Diarrhea, Cough, Wounds",
                    "uses": "Rich in vitamin C, boosts immunity, and aids digestion."
                },
                "Punica Granatum (Pomegranate)": {
                    "common_name": "Pomegranate",
                    "scientific_name": "Punica Granatum",
                    "part_used": "Fruits, Peel",
                    "cures":"Anemia, Diarrhea",
                    "uses": "Rich in antioxidants, good for heart health."
                },
                "Santalum Album (Sandalwood)": {
                    "common_name": "Sandalwood",
                    "scientific_name": "Santalum Album",
                    "part_used": "Heartwood, Oil",
                    "cures":"Skin Infections, Stress",
                    "uses": "Wood used for its fragrance; has applications in perfumery."
                },
                "Syzygium Cumini (Jamun)": {
                    "common_name": "Jamun",
                    "scientific_name": "Syzygium Cumini",
                    "part_used": "Fruits, Seeds",
                    "cures":"Diabetes, Digestive Issues",
                    "uses": "Fruit consumed fresh; seeds used in traditional medicine."
                },
                "Syzygium Jambos (Rose Apple)": {
                    "common_name": "Rose Apple",
                    "scientific_name": "Syzygium Jambos",
                    "uses": "Fruits are consumed fresh; known for diuretic and cooling properties; used in folk medicine."
                },
                "Tabernaemontana Divaricata (Crape Jasmine)": {
                    "common_name": "Crape Jasmine",
                    "scientific_name": "Tabernaemontana Divaricata",
                    "uses": "Used in traditional medicine to treat pain, inflammation, and skin diseases."
                },
                "Trigonella Foenum-graecum (Fenugreek)": {
                    "common_name": "Fenugreek",
                    "scientific_name": "Trigonella Foenum-graecum",
                    "part_used": "Leaves, Seeds",
                    "cures":"Inflammation, Diabetes",
                    "uses": "Seeds used in cooking and traditional medicine for blood sugar regulation, digestion, and lactation support."
                }
            }

            predicted_index = np.argmax(preds)
            predicted_plant = index[predicted_index]

            # Check if the plant is medicinal
            if predicted_plant in medicinal_plants:
                medicinal_status = "(Medicinal Plant)"
            else:
                medicinal_status = "(Non-Medicinal Plant)"

            # Get plant details or set default message
            details = plant_info.get(predicted_plant, {
                "common_name": "Not Available",
                "scientific_name": "Not Available",
                "part_used": "Not Available",
                "cures": "Not Available",
                "uses": "No detailed information available."
            })

            # Create the final output text
            text = f"{predicted_plant} {medicinal_status}\n" \
                   f"Common Name: {details['common_name']}\n" \
                   f"Scientific Name: {details['scientific_name']}\n" \
                   f"Part Used: {details['part_used']}\n" \
                   f"Cures: {details['cures']}\n" \
                   f"Uses: {details['uses']}"

            return text  # Return the predicted result

        return "Please upload an image."
    
    except Exception as e:
        import traceback
        print("Error:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)