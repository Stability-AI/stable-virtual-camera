import torch
import torch.nn as nn
from packaging import version
from einops import rearrange, repeat

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
        else:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                **kwargs,
            )

class SevaWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=2)


        b = x.shape[0]
        f = x.shape[1]
        x = rearrange(x, "b f c h w -> (b f) c h w")
        dense_y=rearrange(c["plucker"], "b f c h w -> (b f) c h w")

        #TODO: remove
        c = torch.zeros((b, 1, 1024)).type_as(x).to(x.device)
        c = repeat(c, "b 1 c -> (b f) 1 c", f=f)
        t = repeat(t, "b -> (b f)", f=f)

        out = self.diffusion_model(
            x,
            t=t,
            y=c, # c["crossattn"]
            dense_y=dense_y,
            num_frames=f,
            **kwargs,
        )
        out = rearrange(out, "(b f) c h w -> b f c h w", f=f)
        return out