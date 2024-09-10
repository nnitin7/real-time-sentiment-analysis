from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['text']
    vectorized_input = vectorizer.transform([data])
    sentiment = model.predict(vectorized_input)[0]
    
    return render_template('index.html', prediction=f'Sentiment: {sentiment}')

if __name__ == "__main__":
    app.run(debug=True)
