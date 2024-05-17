from flask import Flask, render_template, request
import markovify

app = Flask(__name__)

# Load the text corpus
with open('story.txt') as f:
    text = f.read()

# Build the Markov chain model
text_model = markovify.Text(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Generate a new sentence using the Markov chain model
    generated_text = text_model.make_sentence()

    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
