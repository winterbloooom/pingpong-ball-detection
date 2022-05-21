import torch
from train.loss import *
import json
import cv2


class Demo:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None
        self.json_path = "C:\\Users\\dogu\\Desktop\\PPB Detection\\pingpong-ball-detection\\source\\calibration.json" # TODO 파일 경로 넣기
        self.camera_matrix, self.dist_coeff = self.parse_calibration(self.json_path)
    
    def parse_calibration(self, path):
        with open(path, "r",) as f:
            calibration_json = json.load(f)

        camera_matrix = self.parse_intrinsic_calibration(calibration_json["intrinsic"])
        dist_coeff = calibration_json["extrinsic"]["distortion_coff"][0]
        
        return camera_matrix, np.array(dist_coeff)
        
    def parse_intrinsic_calibration(self, intrinsic):
        fx = intrinsic["fx"]
        fy = intrinsic["fy"]
        cx = intrinsic["cx"]
        cy = intrinsic["cy"]
        camera_matrix = np.zeros([3, 3], dtype=np.float32)
        camera_matrix[0][0] = fx
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = fy
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1.0

        return camera_matrix


    def run(self):
        answers=[]
        distance_answers = []
        #file_txt = "..\\datasets\\test\\ImageSets\\test.txt"
        file_txt = "C:\\Users\\dogu\\Desktop\\PPB Detection\\pingpong-ball-detection\\source\\yolov3-pytorch\\datasets\\test2\\ImageSets\\test.txt"
        img_names = []
        img_data = []

        with open(file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            img_data.append(i+".jpg")
        dist_bool = True

        for i, (batch, img_name) in enumerate(zip(self.data_loader,img_names)):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, _, _ = batch
            
            #drawBox(input_img.detach().numpy()[0])
            #np.save("torch_input.npy",input_img.detach().numpy())
            
            input_img = input_img.to(self.device, non_blocking=True)
            #show_img = input_img[b].detach().cpu().numpy()
            input_img=input_img.detach().cpu().numpy()
            print(input_img.shape)
        
            input_img= np.transpose(input_img[0],(1,2,0))
            # input_img= np.transpose(input_img[0],(2,1,0))
            if dist_bool:
                input_img = cv2.undistort(input_img, self.camera_matrix, self.dist_coeff, None)
            # 416 416 3

            #1 3 416 416
            input_img= np.transpose(input_img,(2,0,1))
            # input_img= np.transpose(input_img,(2,1,0))

            input_img = torch.from_numpy(np.array([input_img]))
            input_img = input_img.to(self.device, non_blocking=True)
            num_batch = input_img.shape[0]

            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output,
                                                    conf_thres=0.1,
                                                    iou_thres=0.5)
                
                for b in range(num_batch):
                #for b, img_name in enumerate(img_names):
                    if best_box_list[b] is None:
                        continue
                    #print(best_box_list[b])
                    final_box_list = [bbox for bbox in best_box_list[b] if bbox[4] > 0]
                    #print("final :", final_box_list)

                    if final_box_list is None:
                        continue
                    show_img = input_img[b].detach().cpu().numpy()
                    drawBoxlist(show_img, final_box_list, mode=1, name = str(i)+"_"+str(b))
                    print(final_box_list)
                    for ans_list in save_csv(final_box_list, name = img_name):
                        answers.append(ans_list)
                    for ans_list in predict_distance(final_box_list, img_name):
                        distance_answers.append(ans_list)

        pd_answer=pd.DataFrame(answers)
        pd_answer.to_csv("D:/temp/test_img/answer.csv",header=None, index=None)

        pd_dist_answer=pd.DataFrame(distance_answers)
        pd_dist_answer.to_csv("D:/temp/test_img3/answer_dist.csv", header=None, index=None)
