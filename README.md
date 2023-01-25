# Ping-pong Ball Object Detection and Depth Estimation

## Project Pipeline
### Dataset
* Data collection: 자이카에 장착된 카메라를 사용해 탁구공 비디오를 촬영한 뒤, 적절한 시간 간격으로 이미지를 추출해 저장.
* Data labeling: 수집된 데이터에 대해 레이블링 실행

### Model Training
* Data augmentation 진행
* YOLOv3 모델을 파라미터를 조정하며 학습

### Inference, Depth Estimation
1. 자이카에 장착된 카메라(RGB, 어안렌즈)로 이미지를 입력 받는다.
2. 카메라 캘리브레이션 결과를 이용해 왜곡을 제거한 이미지로 변경한다.
3. 모델 추론 결과로 탁구공의 bounding box를 추출한다.
4. Bounding box 정보 및 사전 정보(탁구공 지름, 카메라 높이 등)을 이용해 각 탁구공까지의 거리를 측정한다.

- - -

## Usage
### 사전 준비
* `/source/calibration` : Do camera intrinsic, extrinsic calibration and save the result in json file
* `/source/convert_imgs.py` : 수집한 이미지의 왜곡을 보정해 저장함

### 실제 깊이 추정
`/source/prediction_proejction.py` : Projection method를 이용한 추정 (xycar 카메라로 작동)

![]("./assets/projection.jpg")

`/source/prediction_homography.py` : Prespective method를 이용한 추정 (저장된 이미지 하나로 작동)

![]("./assets/homography.jpg")
- - -

## Further

* 2D 지도에 탁구공의 위치를 표시하기.
* 더 많은 데이터를 모아 탐지 정확도 향상시키기.
* Egde device에서 성능 확보하기: model quantization, model acceleration
* 움직이는 물체의 위치를 추정하기
* MLOps 도입해보기