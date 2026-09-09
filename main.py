import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import time
import os




CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MODEL_PATH = "hand_landmarker.task"



SMOOTHING = 0.25



CLICK_DISTANCE = 40
CLICK_COOLDOWN = 0.5



SWIPE_DISTANCE = 100
SWIPE_COOLDOWN = 0.8



SCROLL_MULTIPLIER = 1.2
SCROLL_MINIMUM = 1
SCROLL_COOLDOWN = 0.015




pyautogui.PAUSE = 0.01




if not os.path.exists(MODEL_PATH):

    print("ERROR: hand_landmarker.task was not found!")
    print("Put hand_landmarker.task in the same folder as main.py")

    exit()




BaseOptions = python.BaseOptions

options = vision.HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.55,

    min_hand_presence_confidence=0.55,

    min_tracking_confidence=0.55
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)




cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

if not cap.isOpened():

    print("Could not access webcam.")

    exit()




screen_width, screen_height = pyautogui.size()




previous_mouse_x = None
previous_mouse_y = None

last_click_time = 0
last_swipe_time = 0
last_scroll_time = 0



swipe_start_x = None



previous_scroll_y = None


timestamp_ms = 0




HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]




while True:

    success, frame = cap.read()

    if not success:

        print("Could not read webcam.")

        break


    
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape


    

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    timestamp_ms += 33


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    gesture = "NO HAND"


    

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        

        points = []

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 255, 255),
                -1
            )


        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


       

        index_up = (
            hand[8].y < hand[6].y
        )

        middle_up = (
            hand[12].y < hand[10].y
        )

        ring_up = (
            hand[16].y < hand[14].y
        )

        pinky_up = (
            hand[20].y < hand[18].y
        )


        

        
        thumb_wrist_distance = math.sqrt(

            (hand[4].x - hand[0].x) ** 2 +

            (hand[4].y - hand[0].y) ** 2
        )


        thumb_open = (
            thumb_wrist_distance > 0.20
        )


       

        index_x = int(
            hand[8].x * w
        )

        index_y = int(
            hand[8].y * h
        )

        thumb_x = int(
            hand[4].x * w
        )

        thumb_y = int(
            hand[4].y * h
        )


        pinch_distance = math.sqrt(

            (index_x - thumb_x) ** 2 +

            (index_y - thumb_y) ** 2
        )


       

        palm_x = int(

            (
                hand[0].x +
                hand[5].x +
                hand[9].x +
                hand[13].x +
                hand[17].x
            )

            / 5

            * w
        )


        palm_y = int(

            (
                hand[0].y +
                hand[5].y +
                hand[9].y +
                hand[13].y +
                hand[17].y
            )

            / 5

            * h
        )


        cv2.circle(
            frame,
            (palm_x, palm_y),
            10,
            (255, 255, 255),
            -1
        )


       
        if (

            thumb_open
            and index_up
            and middle_up
            and ring_up
            and pinky_up

        ):

            gesture = "OPEN HAND - SWIPE"


           

            if swipe_start_x is None:

                swipe_start_x = palm_x


          

            swipe_distance = (

                palm_x -
                swipe_start_x
            )


            current_time = time.time()




            if (

                swipe_distance >
                SWIPE_DISTANCE

                and

                current_time -
                last_swipe_time >
                SWIPE_COOLDOWN

            ):

                pyautogui.hotkey(
                    "ctrl",
                    "tab"
                )


                gesture = "NEXT TAB →"


                last_swipe_time = current_time

                swipe_start_x = None




            elif (

                swipe_distance <
                -SWIPE_DISTANCE

                and

                current_time -
                last_swipe_time >
                SWIPE_COOLDOWN

            ):

                pyautogui.hotkey(
                    "ctrl",
                    "shift",
                    "tab"
                )


                gesture = "← PREVIOUS TAB"


                last_swipe_time = current_time

                swipe_start_x = None


            previous_scroll_y = None




        elif (

            index_up
            and middle_up
            and not ring_up
            and not pinky_up

        ):

            gesture = "TWO FINGER - SCROLL"




            scroll_y = (

                (
                    hand[8].y +
                    hand[12].y
                )

                / 2

                * h
            )


            scroll_x = (

                (
                    hand[8].x +
                    hand[12].x
                )

                / 2

                * w
            )


            scroll_y = int(scroll_y)
            scroll_x = int(scroll_x)


            cv2.circle(
                frame,
                (scroll_x, scroll_y),
                12,
                (255, 255, 255),
                -1
            )


          

            if previous_scroll_y is None:

                previous_scroll_y = scroll_y




            movement = (

                previous_scroll_y -
                scroll_y
            )


            current_time = time.time()


          

            if (

                current_time -
                last_scroll_time

                >

                SCROLL_COOLDOWN
            ):


                
                scroll_value = int(

                    movement *
                    SCROLL_MULTIPLIER
                )


              
                if scroll_value == 0:

                    if movement > 0:

                        scroll_value = SCROLL_MINIMUM

                    elif movement < 0:

                        scroll_value = -SCROLL_MINIMUM

                    else:

                        scroll_value = 0


                
                if scroll_value != 0:

                    pyautogui.scroll(
                        scroll_value
                    )


                    if movement > 0:

                        gesture = "SCROLL ↑"

                    else:

                        gesture = "SCROLL ↓"


                    last_scroll_time = current_time



            previous_scroll_y = scroll_y


            swipe_start_x = None




        elif (

            pinch_distance <
            CLICK_DISTANCE

        ):

            gesture = "CLICK"


            current_time = time.time()


            if (

                current_time -
                last_click_time
                >

                CLICK_COOLDOWN

            ):

                pyautogui.click()


                last_click_time = current_time


            swipe_start_x = None

            previous_scroll_y = None




        elif (

            index_up

            and not middle_up

            and not ring_up

            and not pinky_up

        ):

            gesture = "MOUSE"


            target_x = int(

                hand[8].x *
                screen_width
            )


            target_y = int(

                hand[8].y *
                screen_height
            )




            if previous_mouse_x is None:

                previous_mouse_x = target_x

                previous_mouse_y = target_y




            mouse_x = (

                previous_mouse_x

                +

                (

                    target_x -
                    previous_mouse_x

                )

                * SMOOTHING
            )


            mouse_y = (

                previous_mouse_y

                +

                (

                    target_y -
                    previous_mouse_y

                )

                * SMOOTHING
            )


            pyautogui.moveTo(

                int(mouse_x),

                int(mouse_y)
            )


            previous_mouse_x = mouse_x

            previous_mouse_y = mouse_y


            swipe_start_x = None

            previous_scroll_y = None




        elif (

            not index_up
            and not middle_up
            and not ring_up
            and not pinky_up

        ):

            gesture = "FIST - STOP"


            swipe_start_x = None

            previous_scroll_y = None




        else:

            gesture = "UNKNOWN"


            swipe_start_x = None

            previous_scroll_y = None



    else:

        swipe_start_x = None

        previous_scroll_y = None

        previous_mouse_x = None

        previous_mouse_y = None




    cv2.putText(

        frame,

        gesture,

        (30, 55),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.1,

        (255, 255, 255),

        3
    )




    cv2.putText(

        frame,

        "1 Finger = Mouse | Pinch = Click",

        (30, 95),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        "2 Fingers = FAST Scroll | 5 Fingers = Tabs",

        (30, 125),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    cv2.putText(

        frame,

        "Fist = Stop | Q = Quit",

        (30, 155),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )




    cv2.imshow(

        "Touchless PC Controller",

        frame
    )




    if cv2.waitKey(1) & 0xFF == ord("q"):

        break




cap.release()

cv2.destroyAllWindows()

landmarker.close()

print("Touchless controller stopped.")