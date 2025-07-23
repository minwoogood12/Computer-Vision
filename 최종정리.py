 <<SAM2>>
<training/train.py>
1. if __name__ : config 등록
2. main : cluster 0(로컬) or 1 선택 
3. single_proc_runner() :  GPU 1, 여러개 구분
4. single_proc_run() : trainer객체 생성 및 trainer.run.() 시작

<training/trainer.py>
1. run(self) : train, train_only, val 3가지로 분류 (train_only : run_train())
2. run_train(self) : 
while self.epoch < self.max_epochs:현재 epoch기준 Max_epoch만큼 반복
  dataloader = Iterable(커스텀된 리스트 느낌) = [
    {img_batch: torch.FloatTensor #[T,B,C,H,W]
    obj_to_frame_idx: torch.IntTensor [T, O, 2]
    masks: torch.BoolTensor [T, O, H, W]
    metadata: BatchedVideoMetaData

    dict_key: str}
   ... 
  ] #length : Batch 개수
  outs = self.train_epoch(dataloader) -> 최종 log결과 딕셔너리
이후 로그저장 및, 체크포인터 저장, epoch +1 정도만 실행

3. train_epoch(self, train_loader) : 
모니터링 객체 초기화, phase, loss meter 초기화
for data_iter, batch in enumerate(train_loader) : 전체 배치 수만큼 반복 
  self.run_step(batch, phase, loss_mts, extra_loss_mts) : phase(현재단계 train, val...), loss_mts(loss로그 기록용), extra_loss_mts(부가적 loss 기록용)
  self.scaler.step(self.optim.optimizer) backward에서 계산된 gradient 파라미터 적용
  self.scaler.update() 다음 step을위한 scaler 업데이트
[최종 로그값 딕셔너리 리턴] 
=> 매 배치마다 gradient파라미터 업데이트 (_run_step에서 매 배치마다 backward()) => 즉 배치한번마다 backward, update(step) 실행

4. def _run_step(
        self,
        batch: BatchedVideoDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
loss_dict, batch_size, extra_losses = self._step(batch, self.model, phase,) 이떄 loss_dict : 1개의 값(모든로스의 합) , extra losses(각각 개별 로스)
self.scaler.scale(loss).backward
=>loss_dict만 backward(), extra_losses는 logging용도

5. _step(self,batch: BatchedVideoDatapoint, model: nn.Module, phase: str):
outputs = model(batch) #training/model/sam로 이동 모델 예측 시작
targets = batch.masks #[T,O,H,W] SAM2는 O가 3으로 최대 3개 예측 가능 / SAMURAI는 O가 1로 무조건 단일 예측 모델임
key = batch.dict_key 
loss = self.loss[key](outputs, targets) 로스계산 #outputs = 미정!! targets : [T,O,H,W]
#loss : {"core_loss" : 0.7, "loss_mask" : 0.2 .....}
return core_loss(0.7),  batch_size, (core_loss제외 나머지 로스들)

<training/model/sam2.py>
1. forward(self, input : BatchedVideoDataPoint):
backbone_out = self.forward_image(input.flat_img_batch) #backbone으로 feature 추출 flat_img_batch : [B*T,C,H,W]
backbone_out = self.prepare_prompt_inputs(backbone_out, input) #??
previous_stages_out = self.forward_tracking(backbone_out, iput) #프레임 각각 예측 및 메모리 업데이트   
return previous_stages_out

<sam2/modeling/sam2_base.py>
1.forward_image(self, img_batch: torch.Tensor):
backbone_out = self.image_encoder(img_batch) #backbone처리 image_batch : [B*T, C, H, W]
if self.use_high_res_features_in_sam이 참이면 4개의 레이어중 앞의 2개의 레이어륾 미리 conv연산 처리
return backbone_out #backbone을 거친 feature 리턴ㄱ

<sam2/modeling/backbones/image_encoder.py>
1.forward(self, sample: torch.Tensor):
features, pos = self.neck(self.trunk(sample)) #trunk(vit_encoder-hiera), neck(FPN)
.... 이후 관련 정보 리턴. 여기는 아직 공부 안함

------------일단은 이미지 백본 ------ 마스크 예측까지 생략하고 loss부터해야할듯 싶어요


<training/loss_fpn.py>
- outs_batch[0] keys: ['point_inputs', 'mask_inputs', 'multistep_pred_masks', 'multistep_pred_masks_high_res', 'multistep_pred_multimasks', 'multistep_pred_multimasks_high_res', 'multistep_pred_ious', 'multistep_point_inputs', 'multistep_object_score_logits', 'pred_masks', 'pred_masks_high_res', 'maskmem_features', 'maskmem_pos_enc']
      - point_inputs: type = <class 'NoneType'>
      - mask_inputs: shape = (1, 1, 512, 512), dtype = torch.bool
      - multistep_pred_masks: shape = (1, 1, 128, 128), dtype = torch.float32
      - multistep_pred_masks_high_res: shape = (1, 1, 512, 512), dtype = torch.float32
      - multistep_pred_multimasks: type = <class 'list'>
      - multistep_pred_multimasks_high_res: type = <class 'list'>
      - multistep_pred_ious: type = <class 'list'>
      - multistep_point_inputs: type = <class 'list'>
      - multistep_object_score_logits: type = <class 'list'>
      - pred_masks: shape = (1, 1, 128, 128), dtype = torch.float32
      - pred_masks_high_res: shape = (1, 1, 512, 512), dtype = torch.float32
      - maskmem_features: shape = (1, 64, 32, 32), dtype = torch.bfloat16
      - maskmem_pos_enc: type = <class 'list'>

outs_batch = [{
 multistep_pred_multimasks_high_res :  List[(N,M,H,W)*step]
 multistep_pred_ious : List[B, O] #예측 객체마다 IOU
 multistep_object_score_logits : List[B,1] #이프레임에 객체가 존재할 확률
} , ..... ] #len : 프레임수 
targets_batch : [T, N, H, W]
num_objects = O
1.forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
num_objects = targets_batch의 두번째 차원 (객체 수)
for outs, targets in zip(outs_batch, targets_batch): #8프레임 반복
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items(): #매 프레임 마다 로스값 더하기
                losses[k] += v
return losses 
outputs : {
 multistep_pred_multimasks_high_res :  List[(N,M,H,W)*step] #스탭 개수는 1에서 8로 랜덤 N : 프레임의 객체 개수 M : 각 객체마다 예측된 마스크 수 
 multistep_pred_ious : List[B, O] #예측 객체마다 IOU
 multistep_object_score_logits : List[B,1] #이프레임에 객체가 존재할 확률
}
targets : [N,H,W]
2. _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
target_masks = targets.unsqueeze(1).float() # [N, 1, H, W]
src_masks_list = outputs["multistep_pred_multimasks_high_res"] #List[(N, M, H, W)] length : Step수
ious_list = outputs["multistep_pred_ious"]
object_score_logits_list = outputs["multistep_object_score_logits"]
losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
for src_masks, ious, object_score_logits in zip( #Step수만큼 반복 
            src_masks_list, ious_list, object_score_logits_list
        ):
          self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
return losses

정리 : 
원본 이미지 - [T,B,C,H,W] 하나의 배치 
예측 마스크 - List[(N,M,H,W)) * Step_num] * Frame_num
GT 마스크- [T, N, H, W]
사무라이에서는 1개만 추적하기때문에 N : 1

maskfreevis 
원본이미지 - List[Tensor(C,H,W) *(B*T)]
예측 마스크 - [B, Q, H, W] -> [N,H,W] 매칭 후
GT 마스크-  [N, T,  H, W] - >[N,H,W]
