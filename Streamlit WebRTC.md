# Streamlit WebRTC 

## 개요

Streamlit WebRTC는 Streamlit 애플리케이션에 실시간 비디오 및 오디오 스트리밍 기능을 추가하기 위한 라이브러리이다. WebRTC(Web Real-Time Communication) 프로토콜을 활용하여 웹 브라우저와 Streamlit 서버 간에 실시간으로 데이터를 주고받을 수 있다. 이를 통해 웹캠 비디오 스트리밍, 오디오 처리, 실시간 화상 회의 등의 다양한 기능을 쉽게 구현할 수 있다.

## WebRTC란?

WebRTC는 브라우저 간 실시간 통신을 가능하게 하는 기술 표준이다. 별도의 플러그인 설치 없이 웹 애플리케이션이 음성, 영상, 데이터 등을 실시간으로 전송할 수 있도록 지원한다. WebRTC는 주로 다음과 같은 주요 기능을 제공한다:



### 주요 기능 설명

1. **비디오 프레임 콜백 함수 (`video_frame_callback`)**:
    - `video_frame_callback` 함수는 실시간 비디오 스트림을 처리하는 데 사용. 이 함수는 스트리밍된 비디오 프레임을 `img_container`라는 전역 변수에 저장하여, 이후 다른 처리에 사용할 수 있도록 한다.

2. **비디오 캡처 및 감정 분석 (`capture_and_classify`)**:
    - `capture_and_classify` 함수는 Streamlit WebRTC의 핵심 기능을 활용하여 실시간으로 웹캠 비디오를 캡처하고, 감정 분석을 수행한다. 이 함수는 중립(neutral) 표정과 행복(happy) 표정을 인식할 때까지 웹캠 비디오 스트림을 지속적으로 모니터링하며, 해당 표정이 감지되면 이미지를 캡처하여 저장한다.

    - 이 과정에서 `webrtc_streamer` 함수가 사용되며, 이는 WebRTC 스트림을 시작하고 관리하는 역할을 한다. `video_frame_callback`을 통해 실시간으로 프레임을 처리하면서, 사용자의 표정이 "neutral" 또는 "happy"로 분류될 때까지 대기한다.

### 코드 예시

```python
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_container["img"] = img  # 실시간으로 프레임 저장
    return frame

def capture_and_classify():
    img1 = None
    img2 = None
    ctx = webrtc_streamer(key="camera", video_frame_callback=video_frame_callback, sendback_audio=False)
    
    while ctx.state.playing:
        img = img_container["img"]
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arrimg = Image.fromarray(img_rgb)
        emotion = get_expression(arrimg)
        
        if emotion == 'neutral' and img1 is None:
            img1 = img_rgb
            st.write("Neutral expression captured.")
        elif emotion == 'happy' and img2 is None:
            img2 = img_rgb
            st.write("Happy expression captured.")
            
        if img1 is not None and img2 is not None:
            ctx.state.playing == False
            break
    
    return img1, img2


