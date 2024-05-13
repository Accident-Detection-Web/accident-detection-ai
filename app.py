from flask import Flask, request, render_template, redirect, url_for, flash
import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
import time
from werkzeug.utils import secure_filename
import ffmpeg

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_gps_from_video(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        metadata = next((stream['tags'] for stream in probe['streams'] if 'tags' in stream), None)
        gps_info = metadata.get('location', 'No GPS info available') if metadata else 'No GPS info available'
        return gps_info
    except Exception as e:
        return 'No GPS info available'
    
@app.route('/', methods=['GET', 'POST'])
def upload_file_or_link():
    if request.method == 'POST':
        file = request.files.get('file')
        video_link = request.form.get('video_link')
        gps_link_info = request.form.get('gps_info', '')  # GPS 정보 입력 받기
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            gps_info = extract_gps_from_video(file_path)
        elif video_link:
            gps_info = gps_link_info  # 링크로 받은 GPS 정보 사용
        else:
            flash('No valid video input provided')
            return redirect(request.url)
        
        flash(f'GPS Info: {gps_info}')  # GPS 정보 플래시 메시지로 출력
        return redirect(url_for('upload_file_or_link'))
    return render_template('upload.html')

def process_video(source, model, device):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return ['Failed to open video source'], False

    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 15
    frame_skip = int(fps / target_fps) if fps > 0 else 1
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    frame_count = 0
    results = []
    accident_count = 0
    pause_analysis = False
    resume_time = 0

    while cap.isOpened():
        if pause_analysis:
            if time.time() < resume_time:
                continue
            else:
                pause_analysis = False

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

        if predicted.item() == 0:
            accident_count += 1
            if accident_count >= 3:
                pause_analysis = True
                resume_time = time.time() + 5
                return results, True  # 사고가 3회 이상 감지될 경우 결과와 함께 중지 신호를 반환

        frame_count += 1

    cap.release()
    return results, False  # 정상적으로 분석이 끝난 경우

if __name__ == '__main__':
    app.run(debug=True)
