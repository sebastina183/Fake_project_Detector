from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Existing project database
projects = [
    "Smart Attendance using AI",
    "Plant Disease Prediction",
    "Fake News Detection",
    "Face Recognition Attendance System",
    "Online Voting System"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    user_project = request.form['project']

    all_projects = projects + [user_project]

    # Convert text into vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_projects)

    # Similarity calculation
    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    max_score = similarity.max() * 100

    if max_score > 70:
        result = "Fake / Duplicate Project Detected"
    else:
        result = "Original Project"

    return render_template(
        'result.html',
        project=user_project,
        score=round(max_score, 2),
        result=result
    )

if __name__ == '__main__':
    app.run(debug=True)