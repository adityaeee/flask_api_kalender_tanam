import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods="*")


@app.route("/predict/su", methods=["POST"])
def predictSU():
    try:
        data = request.get_json()
        input = data["data"]
        print("THE DATA:::::", input)
        model = tf.keras.models.load_model("./modelSU.h5")

        result = model.predict([input])
        print("RESULT:::::", result)
        return jsonify({"res": result.tolist()})

    except FileNotFoundError:
        return jsonify({"error": "Model tidak ditemukan"}), 500
    except KeyError:
        return jsonify({"error": "Data input tidak ditemukan"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal melakukan prediksi: {str(e)}"}), 500


@app.route("/predict/ch", methods=["POST"])
def predictCH():
    try:
        data = request.get_json()
        input = data["data"]
        print("THE DATA:::", input)
        model = tf.keras.models.load_model("./modelCHH.h5")

        result = model.predict([input])
        return jsonify({"res": result.tolist()})

    except FileNotFoundError:
        return jsonify({"error": "Model tidak ditemukan"}), 500
    except KeyError:
        return jsonify({"error": "Data input tidak ditemukan"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal melakukan prediksi: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
