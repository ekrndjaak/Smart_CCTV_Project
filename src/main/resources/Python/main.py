import cv2
import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from collections import Counter
from fastapi import FastAPI, Response, Query
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import asyncio
import base64
import os
from datetime import datetime
import httpx

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

keypoint_model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device).eval()
cap = cv2.VideoCapture(0)


async def send_label_and_image_to_spring(label, frame, max_retries=3):
    spring_boot_url = "http://localhost:8080/api/send-label-and-image"

    for attempt in range(max_retries):
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            data = {
                "label": label,
                "image": image_base64
            }

            print(f"전송 시도 {attempt + 1}/{max_retries}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(spring_boot_url, json=data)

                if response.status_code == 200:
                    print(f"라벨과 이미지 전송 성공")
                    return True
                else:
                    print(f"전송 실패 (상태 코드: {response.status_code})")
                    print(f"응답 내용: {response.text}")

        except Exception as e:
            print(f"전송 시도 {attempt + 1} 실패: {str(e)}")
            if attempt < max_retries - 1:
                print(f"{2 ** attempt}초 후 재시도...")
                await asyncio.sleep(2 ** attempt)
            continue

    print("최대 재시도 횟수 초과")
    return False


class MLP(nn.Module):
    def __init__(self, input_size, f1_num, f2_num, f3_num, f4_num, f5_num, f6_num, d1, d2, d3, d4, d5, num_classes):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, f1_num)
        self.fc2 = nn.Linear(f1_num, f2_num)
        self.fc3 = nn.Linear(f2_num, f3_num)
        self.fc4 = nn.Linear(f3_num, f4_num)
        self.fc5 = nn.Linear(f4_num, f5_num)
        self.fc6 = nn.Linear(f5_num, f6_num)
        self.fc7 = nn.Linear(f6_num, num_classes)
        self.dropout1 = nn.Dropout(p=d1)
        self.dropout2 = nn.Dropout(p=d2)
        self.dropout3 = nn.Dropout(p=d3)
        self.dropout4 = nn.Dropout(p=d4)
        self.dropout5 = nn.Dropout(p=d5)

    def forward(self, x):
        out = self.dropout1(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.dropout5(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        return out

first_mlp_model = MLP(12, 64, 128, 256, 256, 128, 64, 0.2, 0.2, 0.2, 0.2, 0.2, 6)
first_mlp_model.load_state_dict(torch.load('Bottom_Loss_Validation_MLP.pth'))
first_mlp_model = first_mlp_model.to(device).eval()

First_MLP_label_map = {0: 'FallDown', 1: 'FallingDown', 2: 'Sit_chair', 3: 'Sit_floor', 4: 'Sleep', 5: 'Stand'}

current_mode = "1"
Label_List = []
nNotDetected = 0
boolFallCheck = False
MLP_Label = ""
box_color = (0,0,0)

#이미지 전처리
def preprocess(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)


#두 점을 이어주는 선 기울기 구하기
def calculate_angle(point1, point2):
    if point1[0] - point2[0] != 0:
        slope = (point1[1] - point2[1]) / (point1[0] - point2[0])
    else:
        slope = 0
    return slope


#키포인트들 각도 텐서로 변경
def make_angle_to_tensor_list(keypoint):
    angles = [calculate_angle(keypoint[i], keypoint[j]) for i, j in [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]]

    angles_tensor = torch.tensor(angles, dtype=torch.float32).unsqueeze(0).to(device)
    return angles_tensor


#박스랑 라벨 그려주기
def draw_box_and_label(frame, boxes, box_color, label):
    x1, y1, x2, y2 = map(int, boxes)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                            1, 2)
    cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), box_color,
                  cv2.FILLED)
    cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    return frame


#신체 스켈레톤 그리기
def draw_body_line(frame, keypoints):
    cnt = 0
    for point in keypoints:
        if cnt >= 5:
            x, y = map(int, point[:2])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cnt += 1

    connections = [
        (5, 6), (5, 11), (6, 12), (11, 12),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for connection in connections:
        start_point = tuple(map(int, keypoints[connection[0]][:2]))
        end_point = tuple(map(int, keypoints[connection[1]][:2]))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    head_x = int((keypoints[3, 0] + keypoints[4, 0]) / 2)
    head_y = int((keypoints[3, 1] + keypoints[4, 1]) / 2)
    cv2.circle(frame, (head_x, head_y), 10, (0, 0, 255), -1)

    return frame


def reset_state():
    global Label_List, nNotDetected, boolFallCheck, MLP_Label, box_color
    Label_List = []
    nNotDetected = 0
    boolFallCheck = False
    MLP_Label = ""
    box_color = (0,0,0)


reset_state()


#1,3번 모드
def One_Person_Detection(frame, output, index, nNotDetected):
    global boolFallCheck, box_color, MLP_Label
    keypoints = output['keypoints'][index[0]].cpu().numpy()
    keypoint_scores = output['keypoints_scores'][index[0]].cpu().numpy()
    boxes = output['boxes'][index[0]].cpu().numpy()

    if sum(keypoint_scores < 0.9) < 2:

        angles_tensor = make_angle_to_tensor_list(keypoints)

        with torch.no_grad():
            prediction = first_mlp_model(angles_tensor)
            predicted_label = torch.max(prediction, 1)[1].item()

        MLP_Label = First_MLP_label_map[predicted_label]

        Label_List.append(predicted_label)
        if len(Label_List) > 10:
            Label_List.pop(0)

        if len(Label_List) == 10:
            if Label_List[-1] == 0:
                counterBefore = Counter(Label_List)
                most_common_count_Before = counterBefore.most_common(1)[0][1]
                counterBeforeLabel = counterBefore.most_common(1)[0][0]

                if most_common_count_Before >= 7 and (counterBeforeLabel == 1 or counterBeforeLabel == 4):
                    box_color = (0, 0, 255)
                    boolFallCheck = True
                    Label_List.clear()
                elif 1 in Label_List and nNotDetected >= 4:
                    box_color = (0, 0, 255)
                    boolFallCheck = True
                    Label_List.clear()
            else:
                box_color = (0, 255, 0)

            frame = draw_box_and_label(frame, boxes, box_color, MLP_Label)

            nNotDetected = 0
    else:
        nNotDetected = min(nNotDetected + 1, 5)

    return frame, boolFallCheck, nNotDetected, MLP_Label


#2,4번 모드
def Several_Person_Detection(frame, outputs, nNotDetected):
    global boolFallCheck, box_color, MLP_Label
    boolNotDetec = False
    for idx in range(len(outputs)):
        output = outputs[idx]
        scores = output['scores'].cpu().numpy()
        high_scores_idx = np.where(scores > 0.95)[0]

        for high_idx in high_scores_idx:
            keypoints = output['keypoints'][high_idx].cpu().numpy()
            keypoint_scores = output['keypoints_scores'][high_idx].cpu().numpy()
            boxes = output['boxes'][high_idx].cpu().numpy()

            if sum(keypoint_scores < 0.9) < 2:

                angles_tensor = make_angle_to_tensor_list(keypoints)

                with torch.no_grad():
                    prediction = first_mlp_model(angles_tensor)
                    predicted_label = torch.max(prediction, 1)[1].item()

                MLP_Label = First_MLP_label_map[predicted_label]

                if MLP_Label == 0 or MLP_Label == 4:
                    box_color = (0, 0, 255)
                    boolFallCheck = True
                else:
                    box_color = (0, 255, 0)

                frame = draw_box_and_label(frame, boxes, box_color, MLP_Label)
                frame = draw_body_line(frame, keypoints)

                nNotDetected = 0
                boolNotDetec = False

            else:
                boolNotDetec = True

    if boolNotDetec == True:
        nNotDetected = min(nNotDetected + 1, 5)

    return frame, boolFallCheck, nNotDetected, MLP_Label


async def generate_frames():
    global Label_List, nNotDetected, boolFallCheck
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다. 1초 후 다시 시도합니다.")
            await asyncio.sleep(1)
            continue

        #이미지 전처리
        input_tensor = preprocess(frame)

        #키포인트 리턴
        with torch.no_grad():
            outputs = keypoint_model(input_tensor)

        #키포인트 추출이 끝났으니 3,4번 모드이면 검정 프레임으로 교체
        if current_mode == "3" or current_mode == "4":
            frame = np.zeros_like(frame)

        #감지 안되었을 때 nNotDetected 증가
        if len(outputs) != 0:
            #1,3일때 한명만 감지, 2,4 일때 다중 감지
            if current_mode == "1" or current_mode == "3":
                output = outputs[0]
                scores = output['scores'].cpu().numpy()
                high_scores_idx = np.where(scores > 0.95)[0]
                if len(high_scores_idx) > 0:
                    frame, boolFallCheck, nNotDetected, Label = One_Person_Detection(frame, output, high_scores_idx, nNotDetected)

            elif current_mode == "2" or current_mode == "4":
                frame, boolFallCheck, nNotDetected, Label = Several_Person_Detection(frame, outputs, nNotDetected)

            #낙상 감지하면 프레임 날리기
            if boolFallCheck == True:
                success = await send_label_and_image_to_spring(Label, frame)
                if not success:
                    print("위험 감지 알림 전송 실패")
                else:
                    print("위험 감지 알림 전송 성공")
                boolFallCheck = False

        else:
            nNotDetected = min(nNotDetected + 1, 5)

        # 화면 송출할 이미지 설정
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')






@app.get("/")
async def index():
    return Response(content="""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <title>헬퍼 메인 페이지</title>
    <style>
        .cctv-video {
            background-color: #e9ecef;
            height: 610px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: #6c757d;
            position: relative;
            border-radius: 10px;
        }
        .btn-custom1, .btn-custom2 {
            width: 100%;
            height: 180px;
            margin-top: 10px;
        }
        .mode-button {
            width: 100%; /* 버튼의 넓이를 100%로 설정하여 CCTV 로그 버튼과 동일하게 함 */
            height: 60px; /* 높이 조정 가능 */
            margin-top: 5px;
        }
        .description {
            text-align: center;
            margin-top: 30px;
        }
        .logo {
            font-size: 50px;
            font-weight: bold;
            text-align: left;
            margin-left: 20px;
        }
        .logo a {
            text-decoration: none;
            color: black;
        }
        .icon-container {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }
        .icon-button {
            border: none;
            background: transparent;
            padding: 15px;
            font-size: 30px;
            color: #6c757d;
        }
        .icon-button:hover {
            background: rgba(108, 117, 125, 0.2);
            color: #495057;
        }
        .divider {
            height: 1px;
            background-color: #ccc;
            margin: 20px 0;
        }
    </style>
</head>
<body>

<div class="container text-center">
    <!-- 헤더 부분 -->
    <div class="row mt-4">
        <div class="col-md-8">
            <div class="logo"><a href="Main.html">HELPER</a></div>
        </div>
        <div class="col-md-4">
            <div class="icon-container">
                <a href="/myPage" class="btn btn-outline-info btn-sm icon-button">
                    <i class="fas fa-user"></i>
                </a>
                <a href="/helper" class="btn btn-outline-warning btn-sm icon-button">
                    <i class="fas fa-headset"></i>
                </a>
            </div>
        </div>
    </div>

    <div class="divider"></div> <!-- 회색 줄 추가 -->

    <div class="row mt-4">
        <div class="col-md-9">
            <!-- CCTV 스트림을 표시할 부분 -->
            <div class="cctv-video">
                <img id="videoStream" src="http://localhost:8000/video_feed" alt="CCTV Stream" style="width: 100%; height: 100%; object-fit: cover;">
            </div>
        </div>
        <div class="col-md-3">
            <a href="/cctvlog" class="btn btn-outline-primary btn-custom1">CCTV 로그</a>
            <div class="btn-group-vertical">
                <h5>모드 선택</h5>
                <button class="btn btn-outline-secondary mode-button" onclick="selectMode('1')">모드 1</button>
                <button class="btn btn-outline-secondary mode-button" onclick="selectMode('2')">모드 2</button>
                <button class="btn btn-outline-secondary mode-button" onclick="selectMode('3')">모드 3</button>
                <button class="btn btn-outline-secondary mode-button" onclick="selectMode('4')">모드 4</button>
            </div>
            <a href="/notice" class="btn btn-outline-secondary btn-custom2">
                <li><strong>2024-09-26:</strong> 새로운 기능이 추가되었습니다! 지금 확인해보세요.</li>
            </a>
        </div>
    </div>
    <div class="divider"></div> <!-- 회색 줄 추가 -->
    <div class="description">
        <h2>우리집을 지키는 스마트한 HELPER캠</h2>
        <p>마음이 편안합니다.<br>
            직장에서, 여행지에서, 언제 어디서나 집을 확인하세요.<br>
            PC 및 모든 기기에서 집안을 살펴볼 수 있습니다. HELPER은 SD 메모리 카드가 필요 없습니다.</p>
        <button class="btn btn-danger" onclick="onFallDetected()">낙상 감지 시뮬레이션</button>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
<script>
    function selectMode(mode) {
        fetch(`http://localhost:8000/set_mode?mode=${mode}`)
            .then(response => {
                if (response.ok) {
                    console.log(`${mode} 모드로 설정되었습니다.`);
                } else {
                    console.error("모드 설정 실패");
                }
            })
            .catch(error => console.error("오류 발생:", error));
    }

    function onFallDetected() {
        const detectedLabel = "FallDown";
        const boundingBox = [100, 200, 300, 400];
        sendDetectionData(detectedLabel, boundingBox);
    }
</script>

</body>
</html>
""", media_type="text/html")


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/set_mode")
async def set_mode(mode: str = Query(...)):
    global current_mode
    current_mode = mode
    reset_state()
    print(f"현재 모드: {current_mode}")
    return {"message": f"{current_mode} 모드로 설정되었습니다."}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

cap.release()
cv2.destroyAllWindows()

