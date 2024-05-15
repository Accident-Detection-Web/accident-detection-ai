import json
from flask import Flask, request, render_template, redirect, url_for, flash
#from flask_cors import CORS
#from flask_jwt_extended import *
from werkzeug.utils import secure_filename
import os
import subprocess
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
import torch.nn as nn
import time
from datetime import datetime
import base64
import ffmpeg
import re
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

# JWT 매니저 설정
#jwt = JWTManager(app)

# 허용된 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 모델 로드
def load_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

def dms_to_decimal(dms):
    # 정규 표현식을 사용하여 DMS 형식을 추출
    match = re.match(r'(\d+) deg (\d+)\u0027 (\d+\.\d+)" ([NSEW])', dms)
    if not match:
        raise ValueError(f"Invalid DMS format: {dms}")    
    degrees = float(match.group(1))
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)    
    decimal = degrees + (minutes / 60) + (seconds / 3600)
    if direction in ['S', 'W']:
        decimal *= -1    
    return decimal

def extract_gps_data(video_path):
    # GPS 정보 추출을 위한 ExifTool 명령어
    command = ['exiftool', '-GPSLatitude', '-GPSLongitude', '-json', video_path]    
    # subprocess.run을 사용하여 명령어 실행
    result = subprocess.run(command, text=True, capture_output=True)    
    # 결과 확인
    if result.stdout:
        # JSON 데이터를 파싱하여 GPS 정보를 추출
        metadata = json.loads(result.stdout)
        if metadata and 'GPSLatitude' in metadata[0] and 'GPSLongitude' in metadata[0]:
            latitude_dms = metadata[0]['GPSLatitude']
            longitude_dms = metadata[0]['GPSLongitude']
            latitude = dms_to_decimal(latitude_dms)
            longitude = dms_to_decimal(longitude_dms)
            return latitude, longitude    
    # GPS 정보가 없는 경우 0, 0 반환
    print("No GPS data available in the video.")
    return 0, 0

# 비디오 링크 업로드 라우트
# @app.route('/api/v1/public/upload-video', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def upload_file_or_link():
    if request.method == 'POST':
        file = request.files.get('file')
        video_link = request.form.get('video_link')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            gps_info = extract_gps_data(file_path)
        elif video_link:
            gps_info = request.form.get('gps_info', '')
        else:
            flash('No valid video input provided')        
        return redirect(url_for('upload_file_or_link'))    
    return render_template('upload.html')

# database에 데이터 추가
def insert_accident_data(accident_info):
    try:
        # 데이터베이스 연결 설정
        connection = mysql.connector.connect(
            host='127.0.0.1',
            database='capstone_ai',
            user='ai_manager',
            password='0000'
        )
        # 쿼리 실행을 위한 커서 생성
        cursor = connection.cursor()
        # SQL 쿼리 작성
        insert_query = """
        INSERT INTO accidents (image, accident, latitude, longitude, date, sort, severity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        # 데이터 삽입 실행
        cursor.execute(insert_query, (
            accident_info['imagePath'], accident_info['accident'], accident_info['latitude'], 
            accident_info['longitude'], accident_info['date'], accident_info['sort'], accident_info['severity']
        ))
        connection.commit()  # 변경사항 저장
        print("Accident data inserted successfully.")
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")


# 비디오 처리
def process_video(source, model, device, gps_info):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return ['Failed to open video source']

    fps = cap.get(cv2.CAP_PROP_FPS)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    results = []
    folder_path = 'C:\\Capston\\accident-detection-ai\\accident-detection-ai\\img'
    frame_times = []
    accident_count = 0
    frame_skip = 1  # 초기 frame_skip 값

    while cap.isOpened() and accident_count < 5:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad(): 
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                #accident = 'No Accident' if predicted.item() == 1 else 'Accident'.
                accident = 0 if predicted.item() == 1 else 1.
                #if accident == 'Accident':
                if accident == 1:
                    accident_count += 1
                    if(accident_count == 5):
                        #이미지 저장
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        full_path = os.path.join(folder_path, filename)
                        cv2.imwrite(full_path, frame)
                        # 이미지 경로, 사고여부, GPS, 위도, 경도 정보 추가
                        lat, lon = gps_info if gps_info != 'No GPS info available' else ('0', '0')
                        accident_info = {
                            "imagePath": full_path,
                            "accident": accident,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort" : 'NULL',
                            "serverity" : 'NULL'
                        }
                        results.append(accident_info)
                        for accident_info in results:
                            insert_accident_data(accident_info)
                    frame_skip = 1  # 사고가 발생하면 다음 프레임 검사
                else:
                    frame_skip = 5  # 사고가 없으면 5 프레임 후 검사        
        frame_count += 1

    cap.release()
    return results

if __name__ == '__main__':
    model, device = load_model()
    app.run(debug=True)
