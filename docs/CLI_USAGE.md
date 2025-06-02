# :computer: CLI Demo

This cli demo allows you to pass in more options and control the model in a fine-grained way, suitable for power users and academic researchers. An examplar command line looks as simple as

```bash
python demo.py --data_path <data_path> [additional arguments]
```

We discuss here first some key attributes:

- `Procedural Two-Pass Sampling`: We recommend enabling procedural sampling by setting `--use_traj_prior True --chunk_strategy <chunk_strategy>` with `<chunk_strategy>` set according to the type of the task.
- `Resolution and Aspect-Ratio`: Default image preprocessing include center cropping. All input and output are square images of size $576\times 576$. To overwrite, the code support to pass in `--W <W> --H <H>` directly. We recommend passing in `--L_short 576` such that the aspect-ratio of original image is kept while the shortest side will be resized to $576$.

## Task

Before diving into the command lines, we introduce `Task` (specified by `--task <task>`) to bucket different usage cases depending on the data constraints in input and output domains (e.g., if the ordering is available).

|         Task         |  Type of NVS   |         Format of `<data_path>`          | Target Views Sorted? | Input and Target Views Sorted? |    Recommended Usage     |
| :------------------: | :------------: | :--------------------------------------: | :------------------: | :----------------------------: | :----------------------: |
|      `img2img`       |    set NVS     | folder (parsable by `ReconfusionParser`) |         :x:          |              :x:               | evaluation, benchmarking |
|      `img2vid`       | trajectory NVS | folder (parsable by `ReconfusionParser`) |  :white_check_mark:  |       :white_check_mark:       | evaluation, benchmarking |
| `img2trajvid_s-prob` | trajectory NVS |               single image               |  :white_check_mark:  |       :white_check_mark:       |         general          |
|    `img2trajvid`     | trajectory NVS | folder (parsable by `ReconfusionParser`) |  :white_check_mark:  |              :x:               |         general          |

### Format of `<data_path>`

For `img2trajvid_s-prob` task, we are generating a trajectory video following preset camera motions or effects given only one input image, the data format as simple as

```bash
<data_path>/
  ├── scene_1.png
  ├── scene_2.png
  └── scene_3.png
```

For all the other tasks, we use a folder for each scene that is parsable by `ReconfusionParser` (see `seva/data_io.py`). It contains (1) a subdirectory containing all views; (2) `transforms.json` defining the intrinsics and extrinsics (OpenGL convention) for each image; and (3) `train_test_split_*.json` file splitting the input and target views, with `*` indicating the number of the input views.

We provide <a href="https://github.com/Stability-AI/stable-virtual-camera/releases/tag/assets_demo_cli">in this release</a> (`assets_demo_cli.zip`) several examplar scenes for you to take reference from. Target views is available if you the data are from academic sources, but in the case where target views is unavailble, we will create dummy black images as placeholders (e.g., the `garden_flythrough` scene). The general data structure follows

```bash
<data_path>/
├── scene_1/
    ├── train_test_split_1.json # for single-view regime
    ├── train_test_split_6.json # for sparse-veiw regime
    ├── train_test_split_32.json # for semi-dense-view regime
    ├── transforms.json
    └── images/
        ├── image_0.png
        ├── image_1.png
        ├── ...
        └── image_1000.png
├── scene_2
└── scene_3
```

You can specify which scene to run by passing in `--data_items scene_1,scene_2` to run, for example, `scene_1` and `scene_2`.

### Recommended Usage

- `img2img` and `img2vid` are recommended to be used for evaluation and benchmarking. These two tasks are used for the quantitative evalution in our <a href="http://arxiv.org/abs/2503.14489">paper</a>. The data is converted from academic datasets so the groundtruth target views are available for metric computation. Check the [`benchmark`](../benchmark/) folder for detailed splits we organize to benchmark different NVS models.
- `img2vid` requries both the input and target views to be sorted, which is usually not guaranteed in general usage.
- `img2trajvid_s-prob` is for general usage but only for single-view regime and fixed preset camera control.
- `img2trajvid` is the task designed for general usage since it does not need the ordering of the input views. This is the task used in the gradio demo.

Next we go over all tasks and provide for each task an examplar command line.

## `img2img`

```bash
python demo.py \
    --data_path <data_path> \
    --num_inputs <P> \
    --video_save_fps 10
```

- `--num_inputs <P>` is only necessary if there are multiple `train_test_split_*.json` files in the scene folder.
- The above command works for the dataset without trajectory prior (e.g., DL3DV-140). When the trajectory prior is available given a benchmarking dataset, for example, `orbit` trajectory prior for the CO3D dataset, we use the `nearest-gt` chunking strategy by setting `--use_traj_prior True --traj_prior orbit --chunking_strategy nearest-gt`. We find this leads to more 3D consistent results.
- For all the single-view conditioning test scenarios: we set `--camera_scale <camera_scale>` with `<camera_scale>` sweeping 20 different camera scales `0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0`.
- In single-view regime for the RealEstate10K dataset, we find increasing `cfg` is helpful: we additionally set `--cfg 6.0` (`cfg` is `2.0` by default).
- For the evaluation in semi-dense-view regime (i.e., DL3DV-140 and Tanks and Temples dataset) with `32` input views, we zero-shot extend `T` to fit all input and target views in one forward. Specifically, we set `--T 90` for the DL3DV-140 dataset and `--T 80` for the Tanks and Temples dataset.
- For the evaluation on ViewCrafter split (including the RealEastate10K, CO3D, and Tanks and Temples dataset), we find zero-shot extending `T` to `25` to fit all input and target views in one forward is better. Also, the V split uses the original image resolutions: we therefore set `--T 25 --L_short 576`.

For example, you can run the following command on the example `dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557` with 3 input views:

```bash
python demo.py \
    --data_path /path/to/assets_demo_cli/ \
    --data_items dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557 \
    --num_inputs 3 \
    --video_save_fps 10
```

## `img2vid`

```bash
python demo.py \
    --data_path <data_path> \
    --task img2vid \
    --replace_or_include_input True \
    --num_inputs <P> \
    --use_traj_prior True \
    --chunk_strategy interp \
```

- `--replace_or_include_input True` is necessary here since input views and target views are mutually exclusive, forming a trajectory together in this task, so we need to append back the input views to the generated target views.
- `--num_inputs <P>` is only necessary if there are multiple `train_test_split_*.json` files in the scene folder.
- We use `interp` chunking strategy by default.
- For the evaluation on ViewCrafter split (including the RealEastate10K, CO3D, and Tanks and Temples dataset), we find zero-shot extending `T` to `25` to fit all input and target views in one forward is better. Also, the V split uses the original image resolutions: we therefore set `--T 25 --L_short 576`.

## `img2trajvid_s-prob`

```bash
python demo.py \
    --data_path <data_path> \
    --task img2trajvid_s-prob \
    --replace_or_include_input True \
    --traj_prior orbit \
    --cfg 4.0,2.0 \
    --guider 1,2 \
    --num_targets 111 \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp
```

- `--replace_or_include_input True` is necessary here since input views and target views are mutually exclusive, forming a trajectory together in this task, so we need to append back the input views to the generated target views.
- Default `cfg` should be adusted according to `traj_prior`.
- Default chunking strategy is `interp`.
- Default guider is `--guider 1,2` (instead of `1`, `1` still works but `1,2` is slightly better).
- `camera_scale` (default is `2.0`) can be adjusted according to `traj_prior`. The model has scale ambiguity with single-view input, especially for panning motions. We encourage to tune up `camera_scale` to `10.0` for all panning motions (`--traj_prior pan-*/dolly*`) if you expect a larger camera motion.

## `img2trajvid`

### Sparse-view regime ($P\leq 8$)

```bash
python demo.py \
    --data_path <data_path> \
    --task img2trajvid \
    --num_inputs <P> \
    --cfg 3.0,2.0  \
    --use_traj_prior True \
    --chunk_strategy interp-gt
```

- `--num_inputs <P>` is only necessary if there are multiple `train_test_split_*.json` files in the scene folder.
- Default `cfg` should be set to `3,2` (`3` being `cfg` for the first pass, and `2` being the `cfg` for the second pass). Try to increase the `cfg` for the first pass from `3` to higher values if you observe blurry areas (usually happens for harder scenes with a fair amount of unseen regions).
- Default chunking strategy should be set to `interp-gt` (instead of `interp`, `interp` can work but usually a bit worse).
- The `--chunk_strategy_first_pass` is set as `gt-nearest` by default. So it can automatically adapt when $P$ is large (up to a thousand frames).

### Semi-dense-view regime ($P>9$)

```bash
python demo.py \
    --data_path <data_path> \
    --task img2trajvid \
    --num_inputs <P> \
    --cfg 3.0  \
    --L_short 576 \
    --use_traj_prior True \
    --chunk_strategy interp
```

- `--num_inputs <P>` is only necessary if there are multiple `train_test_split_*.json` files in the scene folder.
- Default `cfg` should be set to `3`.
- Default chunking strategy should be set to `interp` (instead of `interp-gt`, `interp-gt` is also supported but the results do not look good).
- `T` can be overwritten by `--T <N>,21` (X being extended `T` for the first pass, and `21` being the default `T` for the second pass). `<N>` is dynamically decided now in the code but can also be manually updated. This is useful when you observe that there exist two very dissimilar adjacent anchors which make the interpolation in the second pass impossible. There exist two ways:
  - `--T 96,21`: this overwrites the `T` in the first pass to be exactly `96`.
  - `--num_prior_frames_ratio 1.2`: this enlarges T in the first pass dynamically to be `1.2`$\times$ larger.
