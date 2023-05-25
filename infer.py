from inference_VQ_Diffusion import VQ_Diffusion
import sys

input_text = sys.argv[1]
n_images = sys.argv[2]

VQ_Diffusion_model = VQ_Diffusion(config='OUTPUT/pretrained_model/config_text.yaml', path='OUTPUT/pretrained_model/coco_pretrained.pth')

# Inference VQ-Diffusion
VQ_Diffusion_model.inference_generate_sample_with_condition(input_text, truncation_rate=0.86, save_root="RESULT", batch_size=int(n_images))

# # Inference Improved VQ-Diffusion with learnable classifier-free sampling
# VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=1.0, save_root="RESULT", batch_size=1, guidance_scale=3.0)



# # Inference Improved VQ-Diffusion with fast/high-quality inference
# VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=1, infer_speed=0.5) # high-quality inference, 0.5x inference speed
# VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=1, infer_speed=2) # fast inference, 2x inference speed
# # infer_speed shoule be float in [0.1, 10], larger infer_speed means faster inference and smaller infer_speed means slower inference

# # Inference Improved VQ-Diffusion with purity sampling
# VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=0.86, save_root="RESULT", batch_size=1, prior_rule=2, prior_weight=1) # purity sampling

# # Inference Improved VQ-Diffusion with both learnable classifier-free sampling and fast inference
# VQ_Diffusion_model.inference_generate_sample_with_condition("A group of elephants walking in muddy water", truncation_rate=1.0, save_root="RESULT", batch_size=1, guidance_scale=5.0, infer_speed=2) # classifier-free guidance and fast inference