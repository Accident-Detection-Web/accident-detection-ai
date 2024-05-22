import json
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_cors import CORS
from flask_jwt_extended import *
from werkzeug.utils import secure_filename
import os
import subprocess
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121
from io import BytesIO
import torch.nn as nn
import time
from datetime import datetime
import base64
import ffmpeg
import re
import mysql.connector
from mysql.connector import Error
import requests
import json
from datetime import datetime
import boto3
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# AWS 설정 
# S3_IMG_BUCKET = 'capstone-accident-img'
# S3_VIDEO_BUCKET = 'capstone-video'
# S3_KEY = 'AKIA6ODU7LGDAOSEOHO4'
# S3_SECRET = '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'
# # S3 클라이언트 생성
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=S3_KEY,
#     aws_secret_access_key=S3_SECRET,
#     region_name=S3_REGION
# )
s3_client = boto3.client('s3')
S3_IMG_BUCKET = 'capstone-accident-img'
S3_VIDEO_BUCKET = 'capstone-video'
S3_REGION = 'ap-northeast-2'

app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

# JWT 매니저 설정
jwt = JWTManager(app)

# 허용된 파일 확장자 검사
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# DenseNet모델 로드
def load_densenet_model():
    device = torch.device("cpu")
    model = densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('densenet_model50.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# YOLO 모델 로드 및 새로운 클래스 이름 설정
def load_yolo_model_with_new_classes(model_path, class_names_path):
    model = YOLO(model_path)
    with open(class_names_path, "r") as f:
        new_class_names = json.load(f)
    model.model.names = new_class_names
    print("새로운 클래스 이름:")
    print(type(model.model.names), len(model.model.names))
    print(model.model.names)
    return model

# AWS S3에서 파일을 메모리로 직접 로드
def load_video_from_s3_to_tempfile(bucket, key):
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_stream = response['Body']
    
    # 임시 파일 생성 및 파일 스트림 쓰기
    temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')  # 삭제하지 않고, .mp4 확장자 사용
    temp_file.write(file_stream.read())
    temp_file.close()
    return temp_file.name  # 파일 경로 반환

# # database에 데이터 추가
# def insert_accident_data(imagePath, accident_info):
#     try:
#         # 데이터베이스 연결 설정
#         connection = mysql.connector.connect(
#             host='localhost',
#             database='ai_capstone',
#             user='root',
#             password='Abcd123@'
#         )
#         # 쿼리 실행을 위한 커서 생성
#         cursor = connection.cursor()
#         # SQL 쿼리 작성
#         insert_query = """
#         INSERT INTO accidents (image, accident, latitude, longitude, date, sort, severity)
#         VALUES (%s, %s, %s, %s, %s, %s, %s)
#         """
#         # 데이터 삽입 실행
#         cursor.execute(insert_query, (
#             imagePath, accident_info['accident'], accident_info['latitude'], 
#             accident_info['longitude'], accident_info['date'], accident_info['sort'], accident_info['severity']
#         ))
#         connection.commit()  # 변경사항 저장
#         print("Accident data inserted successfully.")
#     except Error as e:
#         print("Error while connecting to MySQL", e)
#     finally:
#         if connection.is_connected():
#             cursor.close()
#             connection.close()
#             print("MySQL connection is closed")

def sendData(imagePath, accident_info):
    # 자바 스프링 부트 서버의 URL
    url = 'http://3.38.60.73:8080/api/accident/receiving-data'    
    # accident_info를 문자열로 변환
    requestDtoStr = json.dumps(accident_info)    
    # 이미지 URL에서 이미지 파일 다운로드
    image_response = requests.get(imagePath)
    if image_response.status_code != 200:
        print("Failed to download image")
        print(f"Status Code: {image_response.status_code}")
        print(f"Response Text: {image_response.text}")
        return
    # 멀티파트 폼 데이터 준비
    files = {
        'image': ('accident.png', image_response.content, 'image/png'),
        'requestDto': (None, requestDtoStr, 'application/json')
    }    
    # POST 요청 보내기
    response = requests.post(url, files=files)    
    if response.status_code == 200:
        print("Data sent successfully")
    else:
        print(f"Failed to send data: {response.status_code}, {response.text}")

    
def process_streaming_link(video_link, densenet_model, yolo_model, device, gps_info):
    cap = cv2.VideoCapture(video_link)
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)), 
        transforms.ToTensor()])
    
    frame_count = 0
    results = []
    accident_count = 0
    frame_skip = 1  # 초기 frame_skip 값
    accident_detected = False  # 사고가 탐지되었는지 여부

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break        
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad(): 
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                accident = 0 if predicted.item() == 1 else 1.
                if accident == 1:
                    accident_count += 1
                    if accident_count == 3 and not accident_detected:
                        accident_detected = True
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        _, img_encoded = cv2.imencode('.png', frame)
                        img_bytes = img_encoded.tobytes()

                        s3_client = boto3.client('s3')
                        s3_client.put_object(
                            Bucket=S3_IMG_BUCKET,
                            Key=filename,
                            Body=img_bytes,
                            ContentType='image/png'
                        )
                        # YOLO 모델로 프레임 분석
                        results_yolo = yolo_model(frame)[0]
                        yolo_class = "unknown"
                        for result in results_yolo.boxes:
                            cls = int(result.cls[0].item())
                            if str(cls) in yolo_model.model.names:
                                yolo_class = yolo_model.model.names[str(cls)]
                                break
                            
                        # 이미지 경로, 사고여부, GPS, 위도, 경도 정보 추가
                        lat, lon = gps_info if gps_info != None else ('0', '0')
                        imagePath = f"https://{S3_IMG_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort": yolo_class,
                            "severity": f"{confidence.item():.2f}"
                        }
                        #db에 내용추가
                        #insert_accident_data(imagePath, accident_info)
                        results.append(accident_info)
                        sendData(imagePath, accident_info)

                        # 사고 탐지 후 5초 동안 대기
                        time.sleep(5)
                        accident_detected = False  # 사고 탐지 상태 초기화
                        accident_count = 0  # 사고 카운트 초기화
                    frame_skip = 1  # 사고가 발생하면 다음 프레임 검사
                else:
                    frame_skip = 5  # 사고가 없으면 5 프레임 후 검사        
        frame_count += 1
    cap.release()
    return results

def process_video(bucket, key, densenet_model, yolo_model, device, gps_info):
    video_path = load_video_from_s3_to_tempfile(bucket, key)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        os.unlink(video_path)  # 임시 파일 삭제
        return ['Failed to open video source']

    fps = cap.get(cv2.CAP_PROP_FPS)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    frame_count = 0
    results = []
    accident_count = 0
    frame_skip = 1

    while cap.isOpened() and accident_count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                output = densenet_model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                accident = 0 if predicted.item() == 1 else 1
                if accident == 1:
                    accident_count += 1
                    if accident_count == 3:
                        filename = f'accident_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                        _, img_encoded = cv2.imencode('.png', frame)
                        img_bytes = img_encoded.tobytes()

                        s3_client = boto3.client('s3')
                        s3_client.put_object(
                            Bucket=S3_IMG_BUCKET,
                            Key=filename,
                            Body=img_bytes,
                            ContentType='image/png'
                        )
                        # YOLO 모델로 프레임 분석
                        results_yolo = yolo_model(frame)[0]
                        yolo_class = "unknown"
                        for result in results_yolo.boxes:
                            cls = int(result.cls[0].item())
                            if str(cls) in yolo_model.model.names:
                                yolo_class = yolo_model.model.names[str(cls)]
                                break
                        lat, lon = gps_info
                        imagePath = f"https://{S3_IMG_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort": yolo_class,
                            "severity": f"{confidence.item():.2f}"
                        }
                        #insert_accident_data(imagePath, accident_info)
                        results.append(accident_info)
                        sendData(imagePath, accident_info)
                        os.unlink(video_path)  # 임시 파일 삭제
                        return results
                else:
                    frame_skip = 5
        frame_count += 1
    cap.release()
    os.unlink(video_path)  # 임시 파일 삭제
    return results

# 비디오 링크 업로드 라우트
@app.route('/api/v1/public/upload-link', methods=['GET', 'POST'])
def upload_link():
    if 'video_link' not in request.json:
        return jsonify({'error': 'No video link provided'}), 400
    video_link = request.json['video_link']
    try:
        densenet_model, device = load_densenet_model()
        yolo_model = load_yolo_model_with_new_classes('YOLOv8_best.pt', 'class_names.json')
        gps_info = request.form.get('gps_info', '')
        results = process_streaming_link(video_link, densenet_model, yolo_model, device, gps_info)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# 비디오 파일 업로드 라우트
@app.route('/api/v1/public/upload-video', methods=['GET', 'POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 파일을 로컬에 저장하는 대신 메모리에서 바로 S3에 업로드
        try:
            # 파일 내용을 읽어 S3에 저장
            s3_client.upload_fileobj(
                file,
                S3_VIDEO_BUCKET,
                filename,
                ExtraArgs={'ContentType': file.content_type}
            )
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        # 파일 URL 구성
        file_url = f"https://{S3_VIDEO_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
        # 모델 로딩 및 동영상 처리
        densenet_model, device = load_densenet_model()
        yolo_model = load_yolo_model_with_new_classes('YOLOv8_best.pt', 'class_names.json')
        #gps_info = request.form.get('gps_info', '')
        gps_info = ('0', '0')
        results = process_video(S3_VIDEO_BUCKET, filename, densenet_model, yolo_model, device, gps_info)  # 동영상 처리 함수 호출
        return jsonify(results)    
    else:
        flash('File not allowed or missing')
        return redirect(request.url)

    
if __name__ == '__main__':
    app.run(debug=True)


# DOCKERHUB_USERNAME:qkrckdgns99@gmail.com
# DOCKERHUB_PASSWORD:hh22125513!
# EC2_HOST:ec2-3-35-233-44.ap-northeast-2.compute.amazonaws.com
# EC2_USER:ec2-user
# EC2_KEY:-----BEGIN RSA PRIVATE KEY-----
# MIIEpAIBAAKCAQEAt72Z1lDvpeO7sHvU4g5q0Ox1eTwFvrC0+7WLZsXCmF0Oacfu
# GyHQflgqMjxyzQCnMfXsgGkMrk8Jg+zSAIDCHW2OGnukDcJ3NwN3SBtJ8UN299xG
# TLevn/LW5jprH9VW/twxqXEgxLmnrNxY7MVST52F1QZaTDCG7XazERV1dJdKM2if
# Ruqx1G+ONBY8UqUc1FNWojr8ZRJvKo0kPPp4ZA+6Gt2hrJs8dezTbR4MsQM2TCpW
# s8I/BkdLlPYC5WaCAyxuDydkzSPdsjJWXaeWSw1wY1QIvBLspWNaxcW8FJArJLQF
# MK2rfsTx23+n5V5Tg2Q+9hBorYA2uahiy6GDDQIDAQABAoIBAQCtpRHt6S+Sp1aJ
# w3285cMtD0s19/O182oXN8s2pU7yj38/mSL9oUdZIBlAwL/93dAk9zU7ZgwF78we
# UYFl2EmbZh4WCSNRnabs5umjy6ZlzExykkod1rqzftx5WFxFCWneElsctz0wrgQ4
# 6UVg/lp7w3Lnj8lml7XsVXGFg7ItK/L1kxn9xhvYACC5YzfLvnjTRG7nKZV85bZ4
# NqHM8of7kRmNK4YcC6JYvYuz2aVzfecDYNlUktpg3bxs1IofcJspqbP1MZKSeXhF
# octFB0zydw4WEXC39gjxN3ZlasfQUEn4S5ubf+6v7MHqiCILcHXWDk+KmefjjS6A
# /Ij2SiBBAoGBAOUmExvHNSMWQJQnDUG/eZOMg894aI+jgp7mwqniAk+VL8k7nWS7
# wpE+mGFvArQi4aZ4RNkavFeLkPN98TW6RCZOHhQYpwZK4lHkx075FGMrcvjrDZ+R
# lt8HO9UunJ5qB/G8trkLiRMsMGoppUsjda4293kIaYoCspgtuxQHDFs7AoGBAM1F
# Y7voEPq+3IoMfV4b6CDIlR+vC2EGGF0i3mid8ahzxbst7Y769RzQbX3CKxz7PIAm
# IhT0CZyq9KcCP8NdSN6sgTvghcKNDVKq2uv8Vs/zNjUErUkNfZINQXsFqeNKijta
# zlO4kkRmkjUycJBkc+qDO1lNLLZWMstCikQK72ZXAoGAe2QHQhwA9wXSfHSS5OaQ
# Nu2hRKTX2RD5E0u7YvM6v1PcSYX6ePXKDaAhOcYnNIzb6WI14JpO9O1IfnVA3+eP
# Lvk9pSCpP/Au8l45HMNvQP9yh6s6yMQC89HXXDIfUAZUhM1Tr00Q4OwYnfIS+eT8
# R3V9yQTIn/JX7S4i4OPyuWUCgYEAop4ZL9DeOrcfoiHY48g58lbVhL84xYl9nbM0
# /S47NxdYizwMWdxIeKZKR3mejBwgxuju0SivwLTSksg+WXg6dWW2EAiEDyeNaXM/
# cfp7j8x+oivtV9VfKGhl+p73AsCXmAQNNtge0B9uLsSh1lIuXpfOWaXBCUZqgQpa
# 3SLIm4sCgYA9dGf4GIdzAm8Ryzu4DewKy6LUszFOhvIjcFt1JEhg7B+jo5PI7ljr
# 7KKc7FIPwjcv+eSUSz4NLGe7RVxPYFJAJtokOZAhffOEvSEjDgD7U51NTZsZsE0H
# QzVIQY/EOHKZnjMp3TeHom/LXr7vrtoJrG93+gXx0AjTYzmv6dVVjQ==
# -----END RSA PRIVATE KEY-----