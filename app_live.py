from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
import logging
import time

app_live = Flask(__name__)
app_live.config['SECRET_KEY'] = os.urandom(24).hex()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app_live.route('/api/v1/public/upload-link', methods=['POST'])
def upload_link():
    if 'video_link' not in request.form:
        flash('No video link provided')
        return redirect(request.url)
    video_link = request.form['video_link']
    if video_link == '':
        flash('No video link provided')
        return redirect(request.url)
    
    model, device = load_model()
    results = process_video(video_link, model, device)
    results_query = ','.join(results)  # Convert list to comma-separated string for URL passing

    return redirect(url_for('show_results', results=results_query))

@app_live.route('/result')
def show_results():
    try:
        results = request.args.get('results').split(',')
        return render_template('results.html', results=results)
    except Exception as e:
        logging.error(f"Error displaying results: {e}")
        flash('Error displaying results')
        return redirect(url_for('upload_link'))

def load_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

def process_video(video_link, model, device):
    cap = cv2.VideoCapture(video_link)
    if not cap.isOpened():
        logging.error("Failed to open video stream")
        return ['Failed to open video stream']

    start_time = time.time()  # 처리 시작 시간
    max_duration = 3  # 최대 비디오 처리 시간 (초)

    target_fps = 15
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / target_fps) if fps > 0 else 1
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    frame_count = 0
    results = []
    while cap.isOpened():
        current_time = time.time()
        if current_time - start_time > max_duration:  # 최대 지속 시간 확인
            break

        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            accident = 'No Accident' if predicted.item() == 1 else 'Accident'
        results.append(f'Frame {frame_count}: {accident}')
        frame_count += 1

    cap.release()
    return results

if __name__ == '__main__':
    app_live.run(debug=True)
