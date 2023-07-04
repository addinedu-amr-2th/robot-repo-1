# Global Camera System
### 패키지 설명
- Global Camera(CCTV)를 통해 로봇이 인식하지 못하는 범위의 사람을 인식해 로봇의 지도에 반영
- 이를 global path를 생성하는데 반영해, 매장의 손님에게 쾌적한 환경을 제공하고 보다 안정적인 서빙을 도모
  
#### cctv_person_detect
- webcam.py: CCTV 영상을 토픽으로 publish.
- coordinate_publisher.py: 영상 이미지를 subscribe하고, YOLO를 적용하여 사람을 인식. 사람의 좌표를 bounding box를 통해 구하고, 이를 좌표 변환하여 list 형태로 publish.
- coordinate_subscriber.py: 사람 좌표 list를 subscribe.
  

#### 명령어
```
$ ros2 run cctv_person_detect webcam
```  
```
$ ros2 run cctv_person_detect coordinate_publisher
```
```
$ ros2 run cctv_person_detect coordinate_subscriber
```


