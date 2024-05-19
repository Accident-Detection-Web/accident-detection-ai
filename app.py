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

app = Flask(__name__)
CORS(app)

# AWS 설정 
S3_IMG_BUCKET = 'capstone-accident-img'
S3_VIDEO_BUCKET = 'capstone-video'
S3_KEY = 'AKIA6ODU7LGDAOSEOHO4'
S3_SECRET = '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'
S3_REGION = 'ap-northeast-2'
# S3 클라이언트 생성
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=S3_REGION
)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['ALLOWED_EXTENSIONS'] = {'mov', 'mp4', 'avi'}

# JWT 매니저 설정
jwt = JWTManager(app)

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

# database에 데이터 추가
def insert_accident_data(imagePath, accident_info):
    try:
        # 데이터베이스 연결 설정
        connection = mysql.connector.connect(
            host='localhost',
            database='ai_capstone',
            user='root',
            password='Abcd123@'
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
            imagePath, accident_info['accident'], accident_info['latitude'], 
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

    
def process_streaming_link(video_link, model, device, gps_info):
    cap = cv2.VideoCapture(video_link)
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)), 
        transforms.ToTensor()])
    
    frame_count = 0
    results = []
    accident_count = 0
    frame_skip = 1  # 초기 frame_skip 값

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break        
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad(): 
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                accident = 0 if predicted.item() == 1 else 1.
                if accident == 1:
                    accident_count += 1
                    if(accident_count == 5):
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
                        # 이미지 경로, 사고여부, GPS, 위도, 경도 정보 추가
                        lat, lon = gps_info if gps_info != None else ('0', '0')
                        imagePath = f"https://{S3_IMG_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort": "NULL",
                            "severity": "NULL"
                        }
                        #db에 내용추가
                        insert_accident_data(imagePath, accident_info)
                        results.append(accident_info)
                        sendData(imagePath, accident_info)
                    frame_skip = 1  # 사고가 발생하면 다음 프레임 검사
                else:
                    frame_skip = 5  # 사고가 없으면 5 프레임 후 검사        
        frame_count += 1
    cap.release()
    return results

def process_video(bucket, key, model, device, gps_info):
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            input_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
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
                        lat, lon = gps_info if gps_info != None else ('0', '0')
                        imagePath = f"https://{S3_IMG_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
                        accident_info = {
                            "accident": True,
                            "latitude": lat,
                            "longitude": lon,
                            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "sort": "NULL",
                            "severity": "NULL"
                        }
                        insert_accident_data(imagePath, accident_info)
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
        model, device = load_model()
        gps_info = request.form.get('gps_info', '')
        results = process_streaming_link(video_link, model, device, gps_info)
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
            # 파일 URL 구성
            file_url = f"https://{S3_VIDEO_BUCKET}.s3.{S3_REGION}.amazonaws.com/{filename}"
            # 모델 로딩 및 동영상 처리
            model, device = load_model()
            gps_info = request.form.get('gps_info', '')
            results = process_video(S3_VIDEO_BUCKET, filename, model, device, gps_info)  # 동영상 처리 함수 호출
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        flash('File not allowed or missing')
        return redirect(request.url)

    
if __name__ == '__main__':
    model, device = load_model()
    app.run(debug=True)