import subprocess
import json
# command = [
#     'exiftool',
#     '-GPSLatitude=37.51281766925396',
#     '-GPSLongitude=127.09819270196898',
#     'car_car.mp4'
# ]
# subprocess.run(command)

def extract_gps_data(video_path):
    # GPS 정보 추출을 위한 ExifTool 명령어
    command = ['exiftool', '-GPSLatitude', '-GPSLongitude', '-json', video_path]
    
    # subprocess.run을 사용하여 명령어 실행
    result = subprocess.run(command, text=True, capture_output=True)
    
    # 결과 확인
    if result.stdout:
        #print("Extracted GPS Data:", result.stdout)
        metadata = json.loads(result.stdout)
        if metadata and 'GPSLatitude' in metadata[0] and 'GPSLongitude' in metadata[0]:
            latitude = metadata[0]['GPSLatitude']
            longitude = metadata[0]['GPSLongitude']
            print(latitude, longitude)
    else:
        print("No GPS data available in the video.")

# 비디오 파일 경로 지정
video_path = 'uploads/car_car2.mp4'

# 비디오 파일에서 GPS 데이터 추출 및 출력
extract_gps_data(video_path)
