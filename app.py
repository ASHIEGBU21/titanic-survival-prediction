from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("titanic_survival_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        input_data = pd.DataFrame({
            "Pclass": [int(request.form["pclass"])],
            "Sex": [request.form["sex"]],
            "Age": [float(request.form["age"])],
            "Fare": [float(request.form["fare"])],
            "Embarked": [request.form["embarked"]]
        })

        result = model.predict(input_data)

        if result[0] == 1:
            prediction = "🟢 Survived"
        else:
            prediction = "🔴 Did Not Survive"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
