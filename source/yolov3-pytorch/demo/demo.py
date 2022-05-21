import torch
from train.loss import *
import json
import cv2
from util.calibration_parser import *

class Demo:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None
        self.json_path = "C:\\Users\\dogu\\Desktop\\PPB Detection\\pingpong-ball-detection\\source\\calibration.json" # FIXME calibration JSON 파일 경로 넣기
        self.camera_matrix, self.dist_coeff = read_json_file(self.json_path)

    def run(self, mode='None'):
        answers=[]
        # FIXME test file_txt 경로 수정
        #file_txt = "..\\datasets\\test\\ImageSets\\test.txt"
        file_txt = "C:\\Users\\dogu\\Desktop\\PPB Detection\\pingpong-ball-detection\\source\\yolov3-pytorch\\datasets\\test\\ImageSets\\test.txt"
        
        img_names = []

        with open(file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ str(i.replace("\n", ""))+".jpg" for i in f.readlines()]

        for i, (batch, img_name) in enumerate(zip(self.data_loader,img_names)):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, _, _ = batch
        
            input_img = input_img.to(self.device, non_blocking=True)
            num_batch = input_img.shape[0]

            if mode=="dist":
                input_img=input_img.detach().cpu().numpy()
                input_img= np.transpose(input_img[0],(1,2,0))
                input_img = cv2.undistort(input_img, self.camera_matrix, self.dist_coeff, None)
                input_img= np.transpose(input_img,(2,0,1))
                input_img = torch.from_numpy(np.array([input_img]))
            input_img = input_img.to(self.device, non_blocking=True)
            num_batch = input_img.shape[0]

            with torch.no_grad():
                output = self.model(input_img)
                best_box_list = non_max_suppression(output,
                                                    conf_thres=0.1,
                                                    iou_thres=0.5)
                
                for b in range(num_batch):
                    if best_box_list[b] is None:
                        continue
                    final_box_list = [bbox for bbox in best_box_list[b] if bbox[4] > 0]

                    if final_box_list is None:
                        continue
                    show_img = input_img[b].detach().cpu().numpy()
                    drawBoxlist(show_img, final_box_list, mode=1, name = img_name, folder = mode)
                    if mode == "detect":
                        for ans_list in save_csv(final_box_list, name = img_name):
                            answers.append(ans_list)
                    elif mode == "dist":
                        for ans_list in predict_distance(final_box_list, img_name):
                            answers.append(ans_list)

        if mode !="None":
            pd_answer=pd.DataFrame(answers)
            #FIXME csv 저장 경로 수정
            pd_answer.to_csv("D:/temp/test_img_{}/answer_{}.csv".format(str(mode),str(mode)),header=None, index=None)


