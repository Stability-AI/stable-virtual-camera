# :rocket: Gradio Demo

This gradio demo is the simplest starting point for you play with our project.

You can either visit it at our huggingface space [here](https://huggingface.co/spaces/stabilityai/stable-virtual-camera) or run it locally yourself by

```bash
python demo_gr.py
```

We provide two ways to use our demo:

1. `Basic` mode, where user can upload a single image, and set a target camera trajectory from our preset options. This is the most straightforward way to use our model, and is suitable for most users.
2. `Advanced` mode, where user can upload one or multiple images, and set a target camera trajectory by interacting with a 3D viewport (powered by [viser](https://viser.studio/latest)). This is suitable for power users and academic researchers.

### `Basic`

This is the default mode when entering our demo (given its simplicity).

User can upload a single image, and set a target camera trajectory from our preset options. This is the most straightforward way to use our model, and is suitable for most users.

Here is a video walkthrough:

https://github.com/user-attachments/assets/4d965fa6-d8eb-452c-b773-6e09c88ca705

You can choose from 13 preset trajectories that are common for NVS (`move-forward/backward` are omitted for visualization purpose):

https://github.com/user-attachments/assets/b2cf8700-3d85-44b9-8d52-248e82f1fb55

More formally:

- `orbit/spiral/lemniscate` are good for showing the "3D-ness" of the scene.
- `zoom-in/out` keep the camera position the same while increasing/decreasing the focal length.
- `dolly zoom-in/out` move camera position backward/forward while increasing/decreasing the focal length.
- `move-forward/backward/up/down/left/right` move camera position in different directions.

Notes:

- For a 80 frame video at `786x576` resolution, it takes around 20 seconds for the first pass generation, and around 2 minutes for the second pass generation, tested with a single H100 GPU.
- Please expect around ~2-3x more times on HF space.

### `Advanced`

This is the power mode where you can have very fine-grained control over camera trajectories.

User can upload one or multiple images, and set a target camera trajectory by interacting with a 3D viewport. This is suitable for power users and academic researchers.

Here is a video walkthrough

https://github.com/user-attachments/assets/dcec1be0-bd10-441e-879c-d1c2b63091ba

Notes:

- For a 134 frame video at `576x576` resolution, it takes around 16 seconds for the first pass generation, and around 4 minutes for the second pass generation, tested with a single H100 GPU.
- Please expect around ~2-3x more times on HF space.

### Inpainting

The inpainting feature allows you to fill in missing or corrupted parts of a video using corresponding trajectories and masks for each frame. This feature is useful for tasks such as video restoration and editing.

To use the inpainting feature, follow these steps:

1. Prepare your input video, trajectory, and masks for each frame.
2. Run the `demo.py` script with the appropriate arguments to enable inpainting.
3. The script will process the input video and generate the inpainted output.

Example command:

```bash
python demo.py --data_path <data_path> --inpainting --trajectory <trajectory_path> --masks <masks_path>
```

### Pro tips

- If the first pass sampling result is bad, click "Abort rendering" button in GUI to avoid stucking at second pass sampling such that you can try something else.

### Performance benchmark

We have tested our gradio demo in both a local environment and the HF space environment, across different modes and compilation settings. Here are our results:
| Total time (s) | `Basic` first pass | `Basic` second pass | `Advanced` first pass | `Advanced` second pass |
|:------------------------:|:-----------------:|:------------------:|:--------------------:|:---------------------:|
| HF (L40S, w/o comp.) | 68 | 484 | 48 | 780 |
| HF (L40S, w/ comp.) | 51 | 362 | 36 | 587 |
| Local (H100, w/o comp.) | 35 | 204 | 20 | 313 |
| Local (H100, w/ comp.) | 21 | 144 | 16 | 234 |

Notes:

- HF space uses L40S GPU, and our local environment uses H100 GPU.
- We opt-in compilation by `torch.compile`.
- `Basic` mode is tested by generating 80 frames at `768x576` resolution.
- `Advanced` mode is tested by generating 134 frames at `576x576` resolution.
