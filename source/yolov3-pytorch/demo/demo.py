import torch
from train.loss import *

class Demo:
    def __init__(self, model, data, data_loader, device, hparam):
        self.model = model
        self.class_list = data.class_str
        self.data_loader = data_loader
        self.device = device
        self.yololoss = YoloLoss(self.device, self.model.n_classes, hparam['ignore_cls'])
        self.preds = None

    def run(self):
        answers=[]
        #file_txt = "..\\datasets\\test\\ImageSets\\test.txt"
        file_txt = "C:\\Users\\dogu\\Desktop\\PPB Detection\\pingpong-ball-detection\\source\\yolov3-pytorch\\datasets\\test\\ImageSets\\test.txt"
        img_names = []
        img_data = []
        with open(file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            img_data.append(i+".jpg")

        for i, (batch, img_name) in enumerate(zip(self.data_loader,img_names)):
            #drop the invalid frames
            if batch is None:
                continue
            input_img, _, _ = batch
            
            #drawBox(input_img.detach().numpy()[0])
            #np.save("torch_input.npy",input_img.detach().numpy())
            
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

        pd_answer=pd.DataFrame(answers)
        pd_answer.to_csv("D:/temp/test_img/answer.csv",header=None, index=None)

