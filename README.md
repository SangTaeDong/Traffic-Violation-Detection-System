# Traffic-Violation-Detection-System

<aside>
💡 **서비스 내용**

늘어나는 이륜차의 교통법규 위반, 공익신고 폭증으로 인한
경찰의 인력난을 해결하기 위한 프로젝트

전처리, 객체검출, 위반검출 3단계로 나눠 영상을 분석하고 교통법규 위반을 검출

---

**프로젝트 소개 사이트**
[https://aver1001.github.io](https://aver1001.github.io/)

[KakaoTalk_Video_2022-10-27-11-06-10.mp4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ad884a7-b2e0-4603-b361-18741b58b873/KakaoTalk_Video_2022-10-27-11-06-10.mp4)

</aside>

<aside>
<img src="/icons/layers_gray.svg" alt="/icons/layers_gray.svg" width="40px" /> **System**
Python, Pytorch, OpenCv

</aside>

---

AWS GPU COST로 인해  Front,Back 배포 X
프로젝트 소개 사이트만 따로 제작하여 배포

<aside>
<img src="/icons/layers_gray.svg" alt="/icons/layers_gray.svg" width="40px" /> **Front-end**
JavaScript, React, ~~Redux~~

</aside>

<aside>
<img src="/icons/layers_gray.svg" alt="/icons/layers_gray.svg" width="40px" /> **Back-end**
~~Python, Django~~

</aside>

> **Table of contents**
> 
> 
> ---
> 

---

# `주제선정 동기`

![스크린샷 2022-11-05 오후 4.31.10.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/acc6f261-514b-499f-a19f-e4b553474be3/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.31.10.png)

교통법규 위반 **공익신고수는 급격하게 증가**하고있으며
이로인한 **경찰의 인력난**은 날이갈수록 심해지고 있습니다.

![스크린샷 2022-11-05 오후 4.33.06.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c75dd6f0-f537-4ce7-9f60-1cd10cc4321c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.33.06.png)

하지만 이 문제를 **기술적**으로 해결하는 것은  **굉장히 어려운 문제입니다.**
대용량 영상, 각각 다른 영상들의 환경, 데이터의 특수성 등 해결하기 어려운 난제들이 존재합니다.

![스크린샷 2022-11-05 오후 4.37.00.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe06af12-aeb3-4e10-a600-c7c81f2b6a4c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.37.00.png)

하지만 이 문제를 해결하기 위해 다양한 로직을 통해 **문제를 해결**하였습니다.

# `전처리`

![스크린샷 2022-11-05 오후 4.55.34.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/0d6835fb-a1d4-493a-b568-c62bad9e7d46/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.55.34.png)

- **영상 정리**
    - 예외 처리를 위한 동영상 확장자 파일 분리
    - 영상을 순서대로 불러오기 위한 정렬
    - 영상의 관한 정보 저장 (영상 Frame의 크기 …)
- **프레임 추출**
    - 각각의 영상을 랜덤으로 가져온뒤 (10개) 각각의 첫 프레임을 추출
    (이 프레임들을 통해 Color Range, Bird Eyes View Custom 진행)
- **Color Range Custome**
    
    ![스크린샷 2022-11-05 오후 8.59.30.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/34346cc8-a713-4dff-af04-0ecfebb83f70/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_8.59.30.png)
    
    - 각각의 영상의 색감이 다르기 때문에 차선들의 색영역을 조정해주는 단계.
    1. Lane Detect Model의 결과를 통해 차선을 가져온다.
    2. 차선 히스토그램 분석을 통해 색 영역을 분석한다.
    3. 노란색, 흰색 차선의 히스토그램 범위를 결정한다.
- **Bird Eyes View, Free Sapce Custom**
    
    ![스크린샷 2022-11-05 오후 9.02.14.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5fe00139-09a1-4c76-9577-a54839b3d962/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.02.14.png)
    
    - 각각의 화각이 달라 검출에 필요한 도로의 영역만 가져오는 단계
    1. Lane Detect Model의 결과를 통해 차선을 가져온다.
    2. 정지선, 횡단보도를 제외한 차선만 가져온다.
    3. 차선들을 직선으로 변경 후 소실점을 구해준다.
    4. 모든차선(Bottom), 소실점(Top), 차선기울기(Left, Right)를 이용해 FreeSpace 겸 Bird Eyes View를 구한다

# `객체검출`

![스크린샷 2022-11-05 오후 4.56.32.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70e61531-5c70-457d-a3b2-49bb5f4ad9d5/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.56.32.png)

- **Image Resize**
    - 시간복잡도를 줄이기 위해 Frame을 720*940으로 Resize
- **10번째 Frame**
    - 시간 복잡도와 정확성을 동시에 잡기위해 10프레임중 1프레임은 정확성이 높은 로직구현
    - **Cover_Not_Interset**
        - 노란색 차선 좌우의 객체들은 교통법규 위반검출시 필요하지 않은 객체들이기 때문에
        차선의 정보가 있을경우 노란색 차선의 좌우의 이미지를 삭제
    - **Yolo Traffic Light**
        - Yolo를 직접 학습시켜 신호등의 정보를 검출
        - 객체의 임계값을 조정
            - 좌회전, 노란색 신호등의 경우 다른 신호등의 비해 객체가 잘 안잡혔기 때문에 객체의 Detect 임계값을 조정하여 조금 더 잘 잡히도록 조정
        - 신호등의 위치 확대
            - 블랙박스 영상의 신호등이 굉장히 작은 경우가 잦기때문에 이전의 Free Sapce구역 윗부분을 확대하여 신호등을 Detect
        
         
        
    - **MultiTracker**
        - 10프레임중 9프레임의 객체 위치 파악을 위해 OpenCv의 MultiTracker 객체 생성
    - **highAccCheck**
        - Lane Detect Model을 이용하여 차선의 위치를 정확하게 파악
- **나머지 Frame**
    - 시간 복잡도와 정확성을 동시에 잡기위해 10프레임중 9프레임은 속도가 높은 로직구현
    - **MultiTracker.Update**
        - 객체들의 위치를 파악하기 위해 MultiTracker의 위치를 업데이트
    - **_Stopline**
        - 정지선의 위치를 파악하기 위해 허프변환을 통한 정지선 위치 파악
    - **_laneDetect**
        - 차선의 위치를 파악하기위해 이전의 차선좌표 기준으로 차선 위치 파악

# `위반검출`

![스크린샷 2022-11-05 오후 4.56.55.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4c2fd9e2-b756-4546-842e-7d94733e2039/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.56.55.png)

- **Car_State_Update**
    
    ![스크린샷 2022-11-05 오후 9.26.23.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9876efd0-e177-41b4-a214-c80eef01688c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.26.23.png)
    
    - 객체의 State를 정의 후 State Machine 생성하여 객체들의 State 설정
    
    ![스크린샷 2022-11-05 오후 9.28.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d999fe34-455c-4995-a2b3-b2ee3afc7bbf/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.28.12.png)
    
    - 객체 탐지의 오류 보정을 위해 신호등과 차선의 정보를 10프레임중 최빈값으로 설정
- **Vioation_check**
    - 객체의 State를 정의 후 State Machine 생성하여 객체들의 State 설정
- **검출**
    - 교통법규 위반시 영상 저장

# `성능`

![스크린샷 2022-11-05 오후 9.31.56.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/672022dc-aa37-4a00-a1ca-09a4e701f89f/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-11-05_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.31.56.png)

# `성능개선`

# `ETC..`

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/75400b02-0b5d-4c5f-ac91-827d09f84014/Untitled.png)

## **Data Labeling Program**

위반영상을 구하기 어렵기때문에 직접 1TB(300시간)가량의 데이터를 수집하였으며
프레임 단위로 라벨링해야하는 데이터의 특성으로 인해 직접 라벨링 프로그램을 제작하였음.

[https://github.com/aver1001/Data-Labeling-in-Frame-Units](https://github.com/aver1001/Data-Labeling-in-Frame-Units)

## Front/Back End

초기 계획은 React와 Django를 사용하여 전체 시스탬을 구현하는 것이였으나.
AWS의 GPU cost 문제로 인해 업로드하지 못하였음.

[https://github.com/aver1001/CapstoneFrontEnd_React](https://github.com/aver1001/CapstoneFrontEnd_React)

[https://github.com/aver1001/CapstoneBackEnd_Django](https://github.com/aver1001/CapstoneBackEnd_Django)

프로젝트 소개 홈페이지 제작

[https://aver1001.github.io](https://aver1001.github.io/)
