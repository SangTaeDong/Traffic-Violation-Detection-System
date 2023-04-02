### 2022 상명대 소프트웨어공학과
### 캡스톤 디자인 공과대학 🎉대상🎉

    
## 💡서비스 내용

늘어나는 이륜차의 교통법규 위반, 공익신고 폭증으로 인한
경찰의 인력난을 해결하기 위한 프로젝트

전처리, 객체검출, 위반검출 3단계로 나눠 영상을 분석하고 교통법규 위반을 검출

---
## 프로젝트 소개 영상


[![IU(아이유) _ Into the I-LAND](http://img.youtube.com/vi/_z5OozAbpUA/0.jpg)](https://www.youtube.com/watch?v=_z5OozAbpUA) 


**프로젝트 소개 사이트**
[https://aver1001.github.io](https://aver1001.github.io/)



---

# `주제선정 동기`
<img width="1189" alt="스크린샷 2022-11-05 오후 4 31 10" src="https://user-images.githubusercontent.com/69618305/201271167-c0b01ca6-8608-46d2-8295-fc67e0ce9d07.png">


교통법규 위반 **공익신고수는 급격하게 증가**하고있으며
이로인한 **경찰의 인력난**은 날이갈수록 심해지고 있습니다.

<img width="1189" alt="스크린샷 2022-11-05 오후 4 33 06" src="https://user-images.githubusercontent.com/69618305/201271221-14781788-ff8d-45db-afb5-aea509b8e052.png">

하지만 이 문제를 **기술적**으로 해결하는 것은  **굉장히 어려운 문제입니다.**
대용량 영상, 각각 다른 영상들의 환경, 데이터의 특수성 등 해결하기 어려운 난제들이 존재합니다.


<img width="1189" alt="스크린샷 2022-11-05 오후 4 37 00" src="https://user-images.githubusercontent.com/69618305/201271286-8f04bea7-1a9e-4ccd-9594-784f43f2f726.png">

이 문제를 해결하기 위해 다양한 로직을 통해 **문제를 해결** 하였습니다.

# `전처리`

<img width="1189" alt="스크린샷 2022-11-05 오후 4 55 34" src="https://user-images.githubusercontent.com/69618305/201271471-3f33b3e7-ef41-44cd-a12c-4915eb2d3c51.png">


- **영상 정리**
    - 예외 처리를 위한 동영상 확장자 파일 분리
    - 영상을 순서대로 불러오기 위한 정렬
    - 영상의 관한 정보 저장 (영상 Frame의 크기 …)
- **프레임 추출**
    - 각각의 영상을 랜덤으로 가져온뒤 (10개) 각각의 첫 프레임을 추출
    (이 프레임들을 통해 Color Range, Bird Eyes View Custom 진행)
- **Color Range Custome**
    
    <img width="608" alt="스크린샷 2022-11-05 오후 8 59 30" src="https://user-images.githubusercontent.com/69618305/201271641-6bc392a6-bbc3-486c-b018-9347327e9093.png">

    
    - 각각의 영상의 색감이 다르기 때문에 차선들의 색영역을 조정해주는 단계.
    1. Lane Detect Model의 결과를 통해 차선을 가져온다.
    2. 차선 히스토그램 분석을 통해 색 영역을 분석한다.
    3. 노란색, 흰색 차선의 히스토그램 범위를 결정한다.
- **Bird Eyes View, Free Sapce Custom**
    
    <img width="673" alt="스크린샷 2022-11-05 오후 9 02 14" src="https://user-images.githubusercontent.com/69618305/201271707-3226937d-b968-4c4d-b8bc-0e8f326e5d4a.png">

    
    - 각각의 화각이 달라 검출에 필요한 도로의 영역만 가져오는 단계
    1. Lane Detect Model의 결과를 통해 차선을 가져온다.
    2. 정지선, 횡단보도를 제외한 차선만 가져온다.
    3. 차선들을 직선으로 변경 후 소실점을 구해준다.
    4. 모든차선(Bottom), 소실점(Top), 차선기울기(Left, Right)를 이용해 FreeSpace 겸 Bird Eyes View를 구한다

# `객체검출`

<img width="1189" alt="스크린샷 2022-11-05 오후 4 56 32" src="https://user-images.githubusercontent.com/69618305/201271796-8d93049d-9cc0-4e5f-af2c-b06287d00bce.png">


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

<img width="1189" alt="스크린샷 2022-11-05 오후 4 56 55" src="https://user-images.githubusercontent.com/69618305/201271831-d7f6d3aa-f7d1-4699-ad93-dfdfd8bcbb6f.png">


- **Car_State_Update**
    
    <img width="809" alt="스크린샷 2022-11-05 오후 9 26 23" src="https://user-images.githubusercontent.com/69618305/201271885-550115ec-0128-4f07-9206-e6872583dbb0.png">

    
    - 객체의 State를 정의 후 State Machine 생성하여 객체들의 State 설정
    
    <img width="793" alt="스크린샷 2022-11-05 오후 9 28 12" src="https://user-images.githubusercontent.com/69618305/201272122-96903c69-9d8e-4b89-a86b-c4f515752a01.png">

    
    - 객체 탐지의 오류 보정을 위해 신호등과 차선의 정보를 10프레임중 최빈값으로 설정
- **Vioation_check**
    - 객체의 State를 정의 후 State Machine 생성하여 객체들의 State 설정
- **검출**
    - 교통법규 위반시 영상 저장

# `성능`

<img width="502" alt="스크린샷 2022-11-05 오후 9 31 56" src="https://user-images.githubusercontent.com/69618305/201272172-79ea4696-01d6-4a97-b60b-c7e641b93fb0.png">


# `ETC..`

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

#### 모델
https://www.dropbox.com/scl/fo/75lr8oudf7qrja78kodoy/h?dl=0&rlkey=elb4o3xe4xkp787eituxqfmda

전체 파일 다 최상위에 넣어주시면 됩니다

#### Lane Detect Model 출처
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=197

#### 환경
---
Python                       3.8.10

Package                      Version     
---------------------------- ------------
absl-py                      1.0.0       
aiohttp                      3.8.1       
aiosignal                    1.2.0       
albumentations               1.1.0       
argon2-cffi                  21.3.0      
argon2-cffi-bindings         21.2.0      
asttokens                    2.0.5       
astunparse                   1.6.3       
async-timeout                4.0.2       
attrs                        21.4.0      
backcall                     0.2.0       
beautifulsoup4               4.11.1      
bleach                       5.0.0       
cachetools                   5.0.0       
certifi                      2021.10.8   
cffi                         1.15.0      
charset-normalizer           2.0.12      
cycler                       0.11.0      
debugpy                      1.6.0       
decorator                    5.1.1       
defusedxml                   0.7.1       
entrypoints                  0.4         
executing                    0.8.3       
fastjsonschema               2.15.3      
flatbuffers                  2.0         
fonttools                    4.33.3      
frozenlist                   1.3.0       
fsspec                       2022.3.0    
gast                         0.4.0       
google-auth                  2.6.6       
google-auth-oauthlib         0.4.6       
google-pasta                 0.2.0       
grpcio                       1.46.0      
h5py                         3.6.0       
idna                         3.3         
imageio                      2.19.0      
importlib-metadata           4.11.3      
importlib-resources          5.7.1       
ipykernel                    6.13.0      
ipython                      8.3.0       
ipython-genutils             0.2.0       
ipywidgets                   7.7.0       
jedi                         0.18.1      
Jinja2                       3.1.2       
joblib                       1.1.0       
jsonschema                   4.4.0       
jupyter-client               7.3.0       
jupyter-core                 4.10.0      
jupyterlab-pygments          0.2.2       
jupyterlab-widgets           1.1.0       
Keras                        2.0.8       
Keras-Preprocessing          1.1.2       
kiwisolver                   1.4.2       
libclang                     14.0.1      
Markdown                     3.3.6       
MarkupSafe                   2.1.1       
matplotlib                   3.5.2       
matplotlib-inline            0.1.3       
mistune                      0.8.4       
multidict                    6.0.2       
nbclient                     0.6.2       
nbconvert                    6.5.0       
nbformat                     5.4.0       
nest-asyncio                 1.5.5       
networkx                     2.8         
notebook                     6.4.11      
numpy                        1.22.3      
oauthlib                     3.2.0       
opencv-contrib-python        4.6.0.66    
opencv-python                4.5.5.64    
opencv-python-headless       4.5.5.64    
opt-einsum                   3.3.0       
packaging                    21.3        
pandas                       1.4.2       
pandocfilters                1.5.0       
parso                        0.8.3       
pexpect                      4.8.0       
pickleshare                  0.7.5       
Pillow                       9.1.0       
pip                          20.0.2      
prometheus-client            0.14.1      
prompt-toolkit               3.0.29      
protobuf                     3.20.1      
psutil                       5.9.0       
ptyprocess                   0.7.0       
pure-eval                    0.2.2       
pyasn1                       0.4.8       
pyasn1-modules               0.2.8       
pycparser                    2.21        
pyDeprecate                  0.3.2       
Pygments                     2.12.0      
pyparsing                    3.0.8       
pyrsistent                   0.18.1      
python-dateutil              2.8.2       
pytorch-lightning            1.6.3       
pytz                         2022.1      
PyWavelets                   1.3.0       
PyYAML                       6.0         
pyzmq                        22.3.0      
qudida                       0.0.4       
requests                     2.27.1      
requests-oauthlib            1.3.1       
rsa                          4.8         
scikit-image                 0.19.2      
scikit-learn                 1.0.2       
scipy                        1.8.0       
Send2Trash                   1.8.0       
setuptools                   45.2.0      
six                          1.16.0      
soupsieve                    2.3.2.post1 
stack-data                   0.2.0       
tensorboard                  2.9.0       
tensorboard-data-server      0.6.1       
tensorboard-plugin-wit       1.8.1       
tensorflow                   2.7.0       
tensorflow-estimator         2.7.0       
tensorflow-io-gcs-filesystem 0.25.0      
termcolor                    1.1.0       
terminado                    0.13.3      
threadpoolctl                3.1.0       
tifffile                     2022.5.4    
tinycss2                     1.1.1       
torch                        1.9.0+cu111 
torchaudio                   0.9.0       
torchmetrics                 0.8.1       
torchvision                  0.10.0+cu111
tornado                      6.1         
tqdm                         4.64.0      
traitlets                    5.1.1       
typing-extensions            4.2.0       
ujson                        5.2.0       
urllib3                      1.26.9      
wcwidth                      0.2.5       
webencodings                 0.5.1       
Werkzeug                     2.1.2       
wheel                        0.34.2      
widgetsnbextension           3.6.0       
wrapt                        1.14.1      
yarl                         1.7.2       
zipp                         3.8.0
