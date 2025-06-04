<div align="center">
<h1>Stable Virtual Camera: Generative View Synthesis with Diffusion Models</h1>

<a href="https://stable-virtual-camera.github.io"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-gray.svg"></a>
<a href="http://arxiv.org/abs/2503.14489"><img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2503.14489-B31B1B.svg"></a>
<a href="https://stability.ai/news/introducing-stable-virtual-camera-multi-view-video-generation-with-3d-camera-control"><img src="https://img.shields.io/badge/%F0%9F%93%83%20Blog-Stability%20AI-orange.svg"></a>
<a href="https://huggingface.co/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
<a href="https://huggingface.co/spaces/stabilityai/stable-virtual-camera"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Gradio%20Demo-Huggingface-orange"></a>
<a href="https://www.youtube.com/channel/UCLLlVDcS7nNenT_zzO3OPxQ"><img src="https://img.shields.io/badge/%F0%9F%8E%AC%20Video-YouTube-orange"></a>

[Jensen (Jinghao) Zhou](https://shallowtoil.github.io/)\*, [Hang Gao](https://hangg7.com/)\*
<br>
[Vikram Voleti](https://voletiv.github.io/), [Aaryaman Vasishta](https://www.aaryaman.net/), [Chun-Han Yao](https://chhankyao.github.io/), [Mark Boss](https://markboss.me/)
<br>
[Philip Torr](https://eng.ox.ac.uk/people/philip-torr/), [Christian Rupprecht](https://chrirupp.github.io/), [Varun Jampani](https://varunjampani.github.io/)
<br>
<br>
[Stability AI](https://stability.ai/), [University of Oxford](https://www.robots.ox.ac.uk/~vgg/), [UC Berkeley](https://bair.berkeley.edu/)

</div>

<p align="center">
  <img src="assets/spiral.gif" width="100%" alt="Teaser" style="border-radius:10px;"/>
</p>

<p align="center" border-radius="10px">
  <img src="assets/benchmark.png" width="100%" alt="teaser_page1"/>
</p>

# Overview

`Stable Virtual Camera (SEVA)` is a generalist diffusion model for Novel View Synthesis (NVS), generating 3D consistent novel views of a scene, given any number of input views and target cameras.

# :tada: News

- June 2025 - Release v`1.1` model checkpoint.
- March 2025 - `Stable Virtual Camera` is out everywhere.

# :gear: Versions

| Model Version | \#Parameter | Resolution |                                               Download Link                                                |                                           Update Notes                                           |
| :-----------: | :---------: | :--------: | :--------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|     `1.1`     |    1.3B     |    576P    | 🤗 [Huggingface](https://huggingface.co/stabilityai/stable-virtual-camera/blob/main/modelv1.1.safetensors) | Fixing known issues of foreground objects sometimes being detached from the background in v`1.0` |
|     `1.0`     |    1.3B     |    576P    |   🤗 [Huggingface](https://huggingface.co/stabilityai/stable-virtual-camera/blob/main/model.safetensors)   |                                         Initial release                                          |

You can specify the version via, for example, `load_model(..., model_version=1.1)` in the script.

# :wrench: Installation

```bash
git clone --recursive https://github.com/Stability-AI/stable-virtual-camera
cd stable-virtual-camera
pip install -e .
```

Please note that you will need `python>=3.10` and `torch>=2.6.0`.

Check [INSTALL.md](docs/INSTALL.md) for other dependencies if you want to use our demos or develop from this repo.
For windows users, please use WSL as flash attention isn't supported on native Windows [yet](https://github.com/pytorch/pytorch/issues/108175).

# :open_book: Usage

You need to properly authenticate with Hugging Face to download our model weights. Once set up, our code will handle it automatically at your first run. You can authenticate by running

```bash
# This will prompt you to enter your Hugging Face credentials.
huggingface-cli login
```

Once authenticated, go to our model card [here](https://huggingface.co/stabilityai/stable-virtual-camera) and enter your information for access.

We provide two demos for you to interact with `Stable Virtual Camera`.

### :rocket: Gradio demo

This gradio demo is a GUI interface that requires no expert knowledge, suitable for general users. Simply run

```bash
python demo_gr.py
```

For a more detailed guide, follow [GR_USAGE.md](docs/GR_USAGE.md).

### :computer: CLI demo

This cli demo allows you to pass in more options and control the model in a fine-grained way, suitable for power users and academic researchers. An example command line looks as simple as

```bash
python demo.py --data_path <data_path> [additional arguments]
```

For a more detailed guide, follow [CLI_USAGE.md](docs/CLI_USAGE.md).

For users interested in benchmarking NVS models using command lines, check [`benchmark`](benchmark/) containing the details about scenes, splits, and input/target views we reported in the <a href="http://arxiv.org/abs/2503.14489">paper</a>.

# :question: Q&A

- Training script? See issue https://github.com/Stability-AI/stable-virtual-camera/issues/27, https://github.com/Stability-AI/stable-virtual-camera/issues/42. [@nviolante25](https://www.github.com/nviolante25) has made a pull request (https://github.com/Stability-AI/stable-virtual-camera/pull/51) based on the dicussions.
- License for the output? See issue https://github.com/Stability-AI/stable-virtual-camera/issues/26. The output follows the same non-commercial license.

# :books: Citing

If you find this repository useful, please consider giving a star :star: and citation.

```
@article{zhou2025stable,
    title={Stable Virtual Camera: Generative View Synthesis with Diffusion Models},
    author={Jensen (Jinghao) Zhou and Hang Gao and Vikram Voleti and Aaryaman Vasishta and Chun-Han Yao and Mark Boss and
    Philip Torr and Christian Rupprecht and Varun Jampani
    },
    journal={arXiv preprint arXiv:2503.14489},
    year={2025}
}
```
