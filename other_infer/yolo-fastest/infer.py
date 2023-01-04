from darknet_images import *

def detect():

    config_file = "/home/hoang/darknet-master/data/violence/yolo-fastest-1.1_v4.cfg"
    data_file = "/home/hoang/darknet-master/data/violence/violence.data"
    weights = "/home/hoang/darknet-master/yolofastestv1_modv4_94.66.weights"
    thresh = 0.8

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )

    video_path = "Violence_1_drone - 3of8.mp4"
    
    vid_capture = cv2.VideoCapture(video_path)

    while( vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            image, detections = image_detection(
                frame, network, class_names, class_colors, thresh
            )
            cv2.imshow("Frame", image)
            cv2.waitKey(20)
        else:
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    detect()