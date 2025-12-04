1. sam2.py forward(self, input : BatchedVideoDatapoint)
backbone_out = self.forward_image(input.flat_img_batch)
backbone_out = self.prepare_prompt_inputs(backbone_out, input) 몇번째 프레임에 대해서 보정, gt 비교할지 정함
previous_stages_out = self.forward_tracking(backbone_out, input)


2. sam2.py def forward_tracking(self, backbone_out, input: BatchedVideoDatapoint, return_dict=False):
self._prepare_backbone_features(backbone_out) 전체백본 피처 정리
for i in 8:
  current_out = self.track_step()
return

3. sam2.py def track_step()
self._track_step() #최초 디코더 -> sam2_base.py
self._iter_correct_pt_sampling() #수정해당 프레임만 다시 보정 -> sam2.py
self._encode_memory_in_output() #현재프레임 마스크를 메모리에 인코딩 ->  def _encode_new_memory() -> memory_encoder()
return

4. _track_step()
 self._prepare_memory_conditioned_features() -> 메모리뱅크와 현재피처 섞기
 self._forward_sam_heads() -> 핵심 구조
return
