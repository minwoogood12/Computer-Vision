 if self.training:
            k_size = 3 #3
            rs_images = ImageList.from_tensors(images, self.size_divisibility)
            downsampled_images = F.avg_pool2d(rs_images.tensor.float(), kernel_size=4, stride=4, padding=0) #for img in images]
            images_lab = [torch.as_tensor(color.rgb2lab(ds_image[[2, 1, 0]].byte().permute(1, 2, 0).cpu().numpy()), device=ds_image.device, dtype=torch.float32).permute(2, 0, 1) for ds_image in downsampled_images]
            images_lab_sim = [get_images_color_similarity(img_lab.unsqueeze(0), k_size, 2) for img_lab in images_lab] # ori is 0.3, 0.5, 0.7

            images_lab_sim_nei = [get_neighbor_images_patch_color_similarity(images_lab[ii].unsqueeze(0), images_lab[ii+1].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)] # ori dilation is 3
            images_lab_sim_nei1 = [get_neighbor_images_patch_color_similarity(images_lab[ii+1].unsqueeze(0), images_lab[ii+2].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei2 = [get_neighbor_images_patch_color_similarity(images_lab[ii+2].unsqueeze(0), images_lab[ii+3].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei3 = [get_neighbor_images_patch_color_similarity(images_lab[ii+3].unsqueeze(0), images_lab[ii+4].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei4 = [get_neighbor_images_patch_color_similarity(images_lab[ii+4].unsqueeze(0), images_lab[ii+5].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei5 = [get_neighbor_images_patch_color_similarity(images_lab[ii+5].unsqueeze(0), images_lab[ii+6].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei6 = [get_neighbor_images_patch_color_similarity(images_lab[ii+6].unsqueeze(0), images_lab[ii+7].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]
            images_lab_sim_nei7 = [get_neighbor_images_patch_color_similarity(images_lab[ii+7].unsqueeze(0), images_lab[ii].unsqueeze(0), 3, 3) for ii in range(0, len(images_lab), 5)]


