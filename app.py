from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and cleaned data
model = pickle.load(open('RidgeModel.pkl', 'rb'))
data = pd.read_csv('Cleaned_data.csv')

# Route to display the form
@app.route('/')
def home():
    # Extract unique locations from the dataset
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Prepare the feature vector
    input_features = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Perform any necessary preprocessing here
    # Example: Apply one-hot encoding to location if that was done in the training process

    # Predict using the loaded model
    prediction = model.predict(input_features)[0]

    # Render the result on the same page
    return render_template('index.html', prediction_text=f'Predicted House Price: ₹{prediction*100000:,.2f}', locations=sorted(data['location'].unique()))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
