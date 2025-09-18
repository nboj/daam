from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import PIL.Image
import spacy.tokens
import torch
import torch.nn.functional as F

from .evaluate import compute_ioa
from .utils import compute_token_merge_indices, cached_nlp, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap', 'ParsedHeatMap', 'SyntacticHeatMapPair']

def plot_overlay_heat_map(
    im,
    heat_map,
    word: str | None = None,
    out_file: Path | None = None,
    crop: int | None = None,
    color_normalize: bool = True,
    ax: plt.Axes | None = None,
    dpi: int = 300,
    show_colorbar: bool = True,
    cbar_label: str = "Attention",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_ticks: list[float] | None = None,
):
    created_fig = False

    # Figure & axes
    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        if show_colorbar:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.04], wspace=0.02)
            ax  = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
        else:
            gs = fig.add_gridspec(1, 1)
            ax  = fig.add_subplot(gs[0, 0])
            cax = None
        created_fig = True
    else:
        fig = ax.figure
        if show_colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.02)
        else:
            cax = None

    with auto_autocast(dtype=torch.float32):
        im_np = np.asarray(im).copy()
        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im_np    = im_np[crop:-crop, crop:-crop]

        hm_raw = heat_map.squeeze().float().cpu()
        _vmin = float(hm_raw.min().item()) if vmin is None else float(vmin)
        _vmax = float(hm_raw.max().item()) if vmax is None else float(vmax)
        norm  = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

        # For alpha blending
        if color_normalize:
            hm_unit = (hm_raw - hm_raw.min()) / (hm_raw.max() - hm_raw.min() + 1e-8)
        else:
            hm_unit = hm_raw.clamp_(min=0.0, max=1.0)
        hm_unit_np = hm_unit.numpy()

        # Draw heatmap (mappable for colorbar)
        hm_img = ax.imshow(hm_raw.numpy(), cmap=cmap, norm=norm,
                           interpolation="nearest", aspect="auto")

        # Overlay base image with inverse-heat alpha
        base  = im_np.astype(np.float32) / 255.0
        alpha = (1.0 - hm_unit_np)[..., None]
        rgba  = np.concatenate([base, alpha], axis=-1)
        ax.imshow(rgba, interpolation="nearest", aspect="auto")

        # Clean axes
        ax.set_axis_off()

        # TRANSPARENCY: figure + axes (no background patch)
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        if cax is not None:
            cax.set_facecolor('none')

        # Title ABOVE the image (not over it) with transparent background
        if word:
            st = fig.suptitle(word, y=0.99, va="top")
            st.set_bbox(dict(facecolor='none', edgecolor='none', pad=0))
            # Leave a hair of space at the top so title isn't on the image
            fig.subplots_adjust(top=0.95)

        # Colorbar in its own transparent axis (does NOT overlap image)
        if show_colorbar and cax is not None:
            cb = fig.colorbar(hm_img, cax=cax)
            cb.set_label(cbar_label)
            if cbar_ticks is not None:
                cb.set_ticks(cbar_ticks)
            cb.outline.set_visible(False)
            cax.yaxis.set_ticks_position("right")
            cax.yaxis.set_label_position("right")
            # Transparent axis background (only the gradient stays)
            cax.set_facecolor('none')

        # IMPORTANT: do NOT force ax to [0,0,1,1] when the colorbar exists
        # (that would make the image cover the cbar/title area)
        if not show_colorbar:
            ax.set_position([0, 0, 1, 1])  # full-bleed only when no cbar

        if out_file is not None:
            fig.savefig(out_file, bbox_inches="tight", pad_inches=0, transparent=True)

    if created_fig:
        plt.close(fig)

#def plot_overlay_heat_map(im, heat_map, word=None, out_file=None, crop=None, color_normalize=True, ax=None):
#    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
#    if ax is None:
#        plt.clf()
#        plt.rcParams.update({'font.size': 24})
#        plt_ = plt
#    else:
#        plt_ = ax
#
#    with auto_autocast(dtype=torch.float32):
#        im = np.array(im)
#
#        if crop is not None:
#            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
#            im = im[crop:-crop, crop:-crop]
#
#        if color_normalize:
#            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')
#        else:
#            heat_map = heat_map.clamp_(min=0, max=1)
#            plt_.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet', vmin=0.0, vmax=1.0)
#
#        im = torch.from_numpy(im).float() / 255
#        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
#        plt_.imshow(im)
#
#        if word is not None:
#            if ax is None:
#                plt.title(word)
#            else:
#                ax.set_title(word)
#
#        if out_file is not None:
#            plt.savefig(out_file)


class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, out_file=None, color_normalize=True, ax=None, **expand_kwargs):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(
            image,
            self.expand_as(image, **expand_kwargs),
            word=self.word,
            out_file=out_file,
            color_normalize=color_normalize,
            ax=ax
        )

    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


@dataclass
class SyntacticHeatMapPair:
    head_heat_map: WordHeatMap
    dep_heat_map: WordHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class ParsedHeatMap:
    word_heat_map: WordHeatMap
    token: spacy.tokens.Token


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)

    def parsed_heat_maps(self) -> Iterable[ParsedHeatMap]:
        for token in cached_nlp(self.prompt):
            try:
                heat_map = self.compute_word_heat_map(token.text)
                yield ParsedHeatMap(heat_map, token)
            except ValueError:
                pass

    def dependency_relations(self) -> Iterable[SyntacticHeatMapPair]:
        for token in cached_nlp(self.prompt):
            if token.dep_ != 'ROOT':
                try:
                    dep_heat_map = self.compute_word_heat_map(token.text)
                    head_heat_map = self.compute_word_heat_map(token.head.text)

                    yield SyntacticHeatMapPair(head_heat_map, dep_heat_map, token.head.text, token.text, token.dep_)
                except ValueError:
                    pass


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head


class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
