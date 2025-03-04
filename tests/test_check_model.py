import sys

import torch

from stableviews.utils import load_model

sys.path.insert(0, "/admin/home-hangg/projects/stable-research/")
from scripts.threeD_diffusion.run_eval import init_model

device = torch.device("cuda:0")

input_dict = torch.load("tests/model_input_dict.pth", map_location=device)

version_dict, engine = init_model(
    version="prediction_3D_SD21V_discrete_plucker_norm_replace",
    config="/admin/home-hangg/projects/stable-research/configs/3d_diffusion/jensen/inference/sd_3d-view-attn_21FT_discrete_no-clip-txt_pl---nk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784_ckpt600000.yaml",
)
model_sgm = engine.model.diffusion_model
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
):
    output_sgm = model_sgm(**input_dict)
#
# state_dict = safetensors.torch.load_file(
#     "/admin/home-hangg/projects/stable-research/logs/inference/3d_diffusion-jensen-3d_attn-all1_img2vid25_FT21drunk_plucker_concat_norm_mv_cat3d_v3_discrete_no-clip-txt_3d-attn-with-view-attn-mixing_freeze-pretrained_5355784/epoch=000000-step=000600000_inference.safetensors",
#     device=device,
# )
# model_state_dict = {
#     k.removeprefix("model.diffusion_model."): v
#     for k, v in state_dict.items()
#     if k.startswith("model.diffusion_model.")
# }

# model_sgm1 = _3DUNetModelWithViewAttn(
#     unflatten_names=["middle_ds8", "output_ds4", "output_ds2"],
#     in_channels=11,
#     model_channels=320,
#     out_channels=4,
#     num_res_blocks=2,
#     attention_resolutions=[4, 2, 1],
#     dropout=0.0,
#     channel_mult=[1, 2, 4, 4],
#     conv_resample=True,
#     dims=2,
#     num_classes=None,
#     use_checkpoint=False,
#     num_heads=-1,
#     num_head_channels=64,
#     num_heads_upsample=-1,
#     use_scale_shift_norm=False,
#     resblock_updown=False,
#     use_new_attention_order=False,
#     use_spatial_transformer=True,
#     transformer_depth=1,
#     transformer_depth_middle=None,
#     context_dim=1024,
#     time_downup=False,
#     time_context_dim=None,
#     extra_ff_mix_layer=True,
#     use_spatial_context=True,
#     time_block_merge_strategy="fixed",
#     time_block_merge_factor=0.5,
#     spatial_transformer_attn_type="softmax-xformers",
#     time_kernel_size=3,
#     use_linear_in_transformer=True,
#     legacy=False,
#     adm_in_channels=None,
#     use_temporal_resblock=True,
#     disable_temporal_crossattention=False,
#     time_mix_config={"target": "stableviews._sgm_impl.SkipConnect"},
#     time_mix_legacy=True,
#     max_ddpm_temb_period=10000,
#     replicate_time_mix_bug=False,
#     use_dense_embed=True,
#     dense_in_channels=6,
#     use_dense_scale_shift_norm=True,
#     extra_out_layers=0,
#     original_out_channels=4,
# )
# model_sgm1.load_state_dict(model_state_dict)
# model_sgm1.eval().to(device)
# with torch.inference_mode(), torch.autocast("cuda"):
#     output_sgm1 = model_sgm1(**input_dict)

model = load_model(device, verbose=True).eval()
with (
    torch.inference_mode(),
    torch.autocast(device_type=device.type, dtype=torch.bfloat16),
):
    output = model(
        x=input_dict["x"],
        t=input_dict["timesteps"],
        y=input_dict["context"],
        dense_y=input_dict["dense_y"],
        num_frames=input_dict["num_video_frames"],
    )

# assert torch.allclose(output_sgm, output), __import__("ipdb").set_trace()
# assert torch.allclose(output_sgm1, output), __import__('ipdb').set_trace()
__import__("ipdb").set_trace()
