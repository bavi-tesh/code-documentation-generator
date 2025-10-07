import docly
import os
import requests
from PIL import Image
import cv2
import fitz  # PyMuPDF for PDF processing
import spacy
import re
import nltk
from nltk.tokenize import word_tokenize
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import openai

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize necessary transformers and other tools
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained BERT models
classifier_sentiment = pipeline('sentiment-analysis', model= 'bert-base-uncased')
classifier_zero_shot = pipeline('zero-shot-classification')
classifier_emotion = pipeline('text-classification', model='distilbert-base-uncased', return_all_scores=True)

# Initialize OpenAI API
openai.api_key = ''

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Function to preprocess text with SpaCy
def preprocess_text_with_spacy(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Function to extract text from URLs
def extract_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return clean_text(text)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in document])
    return clean_text(text)

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image)

# Function to preprocess videos
def preprocess_videos(video_paths, output_folder, seconds_per_frame=2):
    def create_output_folder(output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def get_video_name(video_path):
        return os.path.splitext(os.path.basename(video_path))[0]

    def save_frame(frame, frame_filename):
        cv2.imwrite(frame_filename, frame)

    def extract_frames_from_video(video_path, output_folder, seconds_per_frame):
        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        video_name = get_video_name(video_path)
        success = True

        while success:
            success, frame = cap.read()
            if success and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (frame_rate * seconds_per_frame) == 0:
                frame_filename = os.path.join(output_folder, f"{video_name}_frame{count}.jpg")
                save_frame(frame, frame_filename)
                count += 1
        cap.release()

    create_output_folder(output_folder)
    for video_path in video_paths:
        extract_frames_from_video(video_path, output_folder, seconds_per_frame)

# Function to analyze sentiment using VADER
def analyze_sentiment_with_vader(text):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    sentiment = {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    }
    return sentiment

# Function to analyze sentiment using TextBlob
def analyze_sentiment_with_textblob(text):
    blob = TextBlob(text)
    sentiment = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }
    return sentiment

# Function to analyze sentiment using BERT
def analyze_sentiment_with_bert(text):
    results = classifier_sentiment(text)
    sentiment = {
        'label': results[0]['label'],
        'score': results[0]['score']
    }
    return sentiment

# Function to detect emotion using BERT
def detect_emotion_with_bert(text):
    results = classifier_emotion(text)
    emotions = {
        'label': results[0]['label'],
        'score': results[0]['score']
    }
    return emotions

# Function to classify text using BERT
def classify_text_with_bert(text, labels):
    results = classifier_zero_shot(text, candidate_labels=labels)
    classification = {
        'label': results['labels'][0],
        'score': results['scores'][0]
    }
    return classification

# Function to generate text using GPT-3
def generate_text_with_gpt3(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=100
    )
    return response['choices'][0]['text']

# Function to classify age appropriateness
def classify_age_appropriateness(texts, labels):
    vectorizer = TfidfVectorizer()
    classifier = MultinomialNB()

    # Train the classifier
    X = vectorizer.fit_transform(texts)
    classifier.fit(X, labels)

    # Example prediction function
    def predict_age_appropriateness(text):
        X_test = vectorizer.transform([text])
        prediction = classifier.predict(X_test)[0]
        return prediction

    return predict_age_appropriateness

# Main function for processing all inputs
def process_all_inputs(image_paths, video_paths, urls, pdf_paths, texts):
    results = {}

    # Preprocess images
    image_results = []
    for image_path in image_paths:
        preprocessed_image = preprocess_image(image_path)
        image_results.append({
            'image_path': image_path,
            'preprocessed_shape': preprocessed_image.shape,
            'preprocessed_sample': preprocessed_image[0, 0, :5].tolist(),  # Sample pixel values for illustration
        })
    results['images'] = image_results

    # Preprocess videos
    output_folder = 'frames/'
    preprocess_videos(video_paths, output_folder)
    results['videos'] = output_folder

    # Process and analyze text from URLs
    url_results = []
    with ThreadPoolExecutor() as executor:
        url_tasks = [executor.submit(extract_text_from_url, url) for url in urls]
        url_texts = [task.result() for task in as_completed(url_tasks)]
    for url, text in zip(urls, url_texts):
        preprocessed_text = preprocess_text_with_spacy(text)
        sentiment_vader = analyze_sentiment_with_vader(text)
        sentiment_textblob = analyze_sentiment_with_textblob(text)
        sentiment_bert = analyze_sentiment_with_bert(text)
        emotion_bert = detect_emotion_with_bert(text)
        age_appropriateness = classify_age_appropriateness([preprocessed_text], ["child-friendly", "teen-friendly", "adult"])
        url_results.append({
            'url': url,
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment_vader': sentiment_vader,
            'sentiment_textblob': sentiment_textblob,
            'sentiment_bert': sentiment_bert,
            'emotion_bert': emotion_bert,
            'age_appropriateness': age_appropriateness
        })
    results['url_texts'] = url_results

    # Process and analyze text from PDFs
    pdf_results = []
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text_with_spacy(text)
        sentiment_vader = analyze_sentiment_with_vader(text)
        sentiment_textblob = analyze_sentiment_with_textblob(text)
        sentiment_bert = analyze_sentiment_with_bert(text)
        emotion_bert = detect_emotion_with_bert(text)
        age_appropriateness = classify_age_appropriateness([preprocessed_text], ["child-friendly", "teen-friendly", "adult"])
        pdf_results.append({
            'pdf_path': pdf_path,
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment_vader': sentiment_vader,
            'sentiment_textblob': sentiment_textblob,
            'sentiment_bert': sentiment_bert,
            'emotion_bert': emotion_bert,
            'age_appropriateness': age_appropriateness
        })
    results['pdf_texts'] = pdf_results

    # Process and analyze provided texts
    text_results = []
    for text in texts:
        preprocessed_text = preprocess_text_with_spacy(text)
        sentiment_vader = analyze_sentiment_with_vader(text)
        sentiment_textblob = analyze_sentiment_with_textblob(text)
        sentiment_bert = analyze_sentiment_with_bert(text)
        emotion_bert = detect_emotion_with_bert(text)
        age_appropriateness = classify_age_appropriateness([preprocessed_text], ["child-friendly", "teen-friendly", "adult"])
        text_results.append({
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment_vader': sentiment_vader,
            'sentiment_textblob': sentiment_textblob,
            'sentiment_bert': sentiment_bert,
            'emotion_bert': emotion_bert,
            'age_appropriateness': age_appropriateness
        })
    results['provided_texts'] = text_results

    return results
image_paths = [
    r"C:\Users\mbavi\Downloads\AD3X\Images\image1.jpg",
    r"C:\Users\mbavi\Downloads\AD3X\Images\image2.jpg",
    r"C:\Users\mbavi\Downloads\AD3X\Images\image3.jpg",
    r"C:\Users\mbavi\Downloads\AD3X\Images\image4.jpg",
    r"C:\Users\mbavi\Downloads\AD3X\Images\image5.jpg"
]
video_paths = [
    r"C:\Users\mbavi\Downloads\AD3X\Videos\video1.mp4",
    r"C:\Users\mbavi\Downloads\AD3X\Videos\video2.mp4",
    r"C:\Users\mbavi\Downloads\AD3X\Videos\video3.mp4",
    r"C:\Users\mbavi\Downloads\AD3X\Videos\video4.mp4",
    r"C:\Users\mbavi\Downloads\AD3X\Videos\video5.mp4"
]
urls = [
    "https://www.alkimi.org/privacy-policy",
    "https://www.wikipedia.org/",
    "https://www.openai.com/",
    "https://www.github.com/",
    "https://www.stackoverflow.com/"
]
pdf_paths = [
    r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_1.pdf",
    r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_2.pdf",
    r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_3.pdf",
    r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_4.pdf",
    r"C:\Users\mbavi\Downloads\AD3X\PDFs\PDF_5.pdf"
]
texts = [
    "Adtech innovations like programmatic advertising have revolutionized how brands target consumers online. By leveraging data analytics and real-time bidding, advertisers can precisely reach their target audiences across multiple digital channels. This efficiency has significantly boosted ROI for many businesses, making ad spending more accountable and results-driven.",
    "As adtech continues to evolve, the debate around data sovereignty and consumer rights intensifies. Companies are grappling with the ethical implications of data collection and targeted advertising, especially in light of recent privacy scandals. Regulators are under pressure to enact stricter laws to protect user data, while advertisers seek innovative ways to maintain relevance without compromising privacyNeutral. The paragraph discusses the ongoing debate and challenges in adtech.",
    "Blockchain technology is being explored as a means to enhance transparency and trust in digital advertising. Proponents believe it can help combat ad fraud and improve accountability. However, the complexity and cost of implementing blockchain solutions are significant barriers to widespread adoption.",
    "Amidst the dimly lit alley, smoke curled lazily from the arded cigarette, casting a haze over the scene. Two figures squared off, their movements sharp and deliberate in the glow of flickering street lamps. The air crackled with tension as fists clenched and insults flew, each word escalating the conflict. It was a raw display of anger and pride, a moment where adrenaline surged and consequences blurred.",
    "Under the canopy of stars, they stood hand in hand, the night alive with whispered secrets and the gentle rustle of leaves. Moonlight painted patterns on the path before them, a silent witness to their quiet exchange. In that fleeting moment, words were unnecessary; their hearts spoke in the language of shared dreams and unspoken promises. It was a scene straight from a romance novel, where time stood still in the embrace of love.""Under the canopy of stars, they stood hand in hand, the night alive with whispered secrets and the gentle rustle of leaves. Moonlight painted patterns on the path before them, a silent witness to their quiet exchange. In that fleeting moment, words were unnecessary; their hearts spoke in the language of shared dreams and unspoken promises. It was a scene straight from a romance novel, where time stood still in the embrace of love."
]

results = process_all_inputs(image_paths, video_paths, urls, pdf_paths, texts)
print(results)
