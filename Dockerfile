# Python 3.9 슬림 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /

# requirements.txt 파일을 컨테이너로 복사
COPY requirements.txt .

# 필요 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 포트 노출
EXPOSE 5000

# 환경 변수 설정
ENV FLASK_APP=app.py

# 애플리케이션 실행
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
