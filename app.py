from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('RidgeModel.pkl', 'rb'))
data = pd.read_csv('Cleaned_data.csv')

@app.route('/')
def home():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    input_features = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(input_features)[0]

    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{prediction*100000:,.2f}', locations=sorted(data['location'].unique()))

if __name__ == "__main__":
    app.run(debug=True, port=5000)