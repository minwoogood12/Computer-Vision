수정한 내용 07.18

1. 기존 sam2.1_hiera_b+_MOSE_finetune.yaml 학습 yaml파일에
"사무라이 구조", "tk, proj, pairwise_loss"추가

-------------------------------------------------------------------------------------

2. trainer.py에 

2-1. color_similarity구하는 코드 추가
def unfold_wo_center(x, kernel_size, dilation):
def unfold_w_center(x, kernel_size, dilation):
def get_images_color_similarity(images, kernel_size, dilation):
def get_neighbor_images_color_similarity(images, images_neighbor, kernel_size, dilation):
def get_neighbor_images_patch_color_similarity(images, images_neighbor, kernel_size, dilation):

2-2. 원본이미지 받아서 color_similarity구해서 loss로 넘겨주는 코드 추가
outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)
        ##추가##
        images = batch.img_batch.squeeze(1)  # shape: [T, B, C, H, W] T : 프레임수
        
        downsampled_images = F.avg_pool2d(images.float(), kernel_size=4, stride=4, padding=0)  # [T, C, H', W']
        images_lab = [
          torch.as_tensor(
              color.rgb2lab(img[[2, 1, 0]].byte().permute(1, 2, 0).cpu().numpy()),
              device=img.device,
              dtype=torch.float32
          ).permute(2, 0, 1)
          for img in downsampled_images
        ]
        images_lab_sim = [get_images_color_similarity(img.unsqueeze(0), 3, 2) for img in images_lab]
        images_lab_sim_nei = get_neighbor_images_patch_color_similarity(images_lab[0].unsqueeze(0), images_lab[1].unsqueeze(0), 3, 3)
        images_lab_sim_nei1 = get_neighbor_images_patch_color_similarity(images_lab[1].unsqueeze(0), images_lab[2].unsqueeze(0), 3, 3)
        images_lab_sim_nei2 = get_neighbor_images_patch_color_similarity(images_lab[2].unsqueeze(0), images_lab[3].unsqueeze(0), 3, 3)
        images_lab_sim_nei3 = get_neighbor_images_patch_color_similarity(images_lab[3].unsqueeze(0), images_lab[4].unsqueeze(0), 3, 3)
        images_lab_sim_nei4 = get_neighbor_images_patch_color_similarity(images_lab[4].unsqueeze(0), images_lab[5].unsqueeze(0), 3, 3)
        images_lab_sim_nei5 = get_neighbor_images_patch_color_similarity(images_lab[5].unsqueeze(0), images_lab[6].unsqueeze(0), 3, 3)
        images_lab_sim_nei6 = get_neighbor_images_patch_color_similarity(images_lab[6].unsqueeze(0), images_lab[7].unsqueeze(0), 3, 3)
        images_lab_sim_nei7 = get_neighbor_images_patch_color_similarity(images_lab[7].unsqueeze(0), images_lab[0].unsqueeze(0), 3, 3)
        ##추가##
      
        key = batch.dict_key  # key for dataset

        
        loss = self.loss[key](outputs, targets,
                              ##추가##
                              images_lab_sim, 
                              images_lab_sim_nei,
                              images_lab_sim_nei2,
                              images_lab_sim_nei3,
                              images_lab_sim_nei4,
                              images_lab_sim_nei5,
                              images_lab_sim_nei6,
                              images_lab_sim_nei7,
                              ##추가##
                              )
 => 기존 maskfreevis는 4까지이지만 sam2는 8개 프레임단위이므로 증가시킴

----------------------------------------------------------------------

3. loss.py에 

3-1. "tk, proj, pairwise_loss"추가했는지 검사하는 코드 추가
if "loss_tk" not in self.weight_dict:
  self.weight_dict["loss_tk"] = 0.0
if "loss_proj" not in self.weight_dict:
  self.weight_dict["loss_proj"] = 0.0
if "loss_pairwise" not in self.weight_dict:
  self.weight_dict["loss_pairwise"] = 0.0
        
3-2. forward()에 images_lab_sim~images_lab_sim_nei7 인수로 받는 코드 추가
def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor,
                ##추가##
                images_lab_sim, 
                images_lab_sim_nei,
                images_lab_sim_nei2,
                images_lab_sim_nei3,
                images_lab_sim_nei4,
                images_lab_sim_nei5,
                images_lab_sim_nei6,
                images_lab_sim_nei7,
                ##추가##
               ):

3-3. loss딕셔너리에 "tk, proj, pairwise_loss" 추가
losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0, "loss_tk" : 0, "loss_proj" : 0, "loss_pairwise" : 0}

3-4. loss_mask_proj에서 "tk, proj, pairwise_loss" 계산
 def loss_masks_proj(self, outputs, targets, num_objects, 
                        images_lab_sim, 
                        images_lab_sim_nei, 
                        images_lab_sim_nei1, 
                        images_lab_sim_nei2, 
                        images_lab_sim_nei3, 
                        images_lab_sim_nei4,
                        images_lab_sim_nei5,
                        images_lab_sim_nei6,
                        images_lab_sim_nei7
                       ):
        loss_tk = 1
        loss_proj = 2
        loss_pairwise = 3

        return loss_tk, loss_proj, loss_pairwise
    ##추가##


3-5. forward 함수 마지막 부분에서 "tk, proj, pairwise_loss"를 계산하고 losses 딕셔너리에 더해줌
##추가##
        loss_tk, loss_proj, loss_pairwise = self.loss_masks_proj(
            src_masks_list, targets_masks, num_objects,
            images_lab_sim,
            images_lab_sim_nei,
            images_lab_sim_nei1,
            images_lab_sim_nei2,
            images_lab_sim_nei3,
            images_lab_sim_nei4,
            images_lab_sim_nei5,
            images_lab_sim_nei6,
            images_lab_sim_nei7
        )
        losses["loss_tk"] += loss_tk
        losses["loss_proj"] += loss_proj
        losses["loss_pairwise"] += loss_pairwise
        loss_sum = loss_tk + loss_proj + loss_pairwise
        losses[CORE_LOSS_KEY] += loss_sum
                   
        ##추가##

3-6. 


3-7. 

3-5. 

