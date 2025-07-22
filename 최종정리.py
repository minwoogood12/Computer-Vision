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
backbone_out = self.forward_image(input.flat_img_batch) #backbone으로 feature 추출
backbone_out = self.prepare_prompt_inputs(backbone_out, input) #??
previous_stages_out = self.forward_tracking(backbone_out, iput) #프레임 각각 예측 및 메모리 업데이트   
return previous_stages_out

