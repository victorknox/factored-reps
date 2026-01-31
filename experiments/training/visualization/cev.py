"""Cumulative explained variance (CEV) visualizations."""

from __future__ import annotations

import re
import warnings
from typing import Any, Mapping

import numpy as np
import plotly.colors
import plotly.graph_objects as go
from omegaconf import DictConfig

from visualization._types import CEVHistory
from visualization.configs import CEVVizConfig
from visualization.styles import (
    ARCHIVAL_COLORSCALE,
    COLORBAR_DEFAULTS,
    TITLE_FONT,
    apply_default_layout,
)


def _empty_figure(title: str, reason: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=reason,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(title=title)
    return fig


def _coerce_cev_config(config: CEVVizConfig | Mapping[str, Any] | DictConfig | None) -> CEVVizConfig:
    if config is None:
        return CEVVizConfig()
    if isinstance(config, CEVVizConfig):
        return config
    return CEVVizConfig.from_dict(config)


# Vertical dual-handle range slider (replaces colorbar)
# Styled with warm archival aesthetic
_DUAL_SLIDER_CSS = """
<style>
.step-slider-container {
  position: absolute;
  right: 10px;
  top: 100px;
  height: 65%;
  width: 70px;
  font-family: Georgia, 'Times New Roman', serif;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.step-slider-title {
  position: absolute;
  right: -18px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 13px;
  color: #333333;
  font-family: Georgia, 'Times New Roman', serif;
  writing-mode: vertical-rl;
  text-orientation: mixed;
  white-space: nowrap;
}
.step-slider-max-label, .step-slider-min-label {
  font-size: 11px;
  color: #666666;
  font-family: Georgia, 'Times New Roman', serif;
  text-align: center;
  min-width: 40px;
}
.step-slider-wrapper {
  position: relative;
  height: 100%;
  width: 34px;
  margin: 5px 0;
}
.step-slider-track {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  top: 0;
  bottom: 0;
  width: 24px;
  background: #e8e4de;
  border-radius: 3px;
}
.step-slider-range {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
  width: 24px;
  background: linear-gradient(to top, #2e3a5f, #2d6a7a, #3a9a8c, #7dbf7d, #d4a84b);
  border-radius: 3px;
}
.step-slider-wrapper input[type="range"] {
  position: absolute;
  left: 0;
  width: 100%;
  height: 100%;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  pointer-events: none;
  margin: 0;
  writing-mode: vertical-lr;
  direction: rtl;
}
.step-slider-wrapper input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 34px;
  height: 5px;
  background: #faf8f5;
  border: 1px solid #aaa;
  border-radius: 2px;
  cursor: ns-resize;
  pointer-events: auto;
  box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}
.step-slider-wrapper input[type="range"]::-moz-range-thumb {
  width: 34px;
  height: 5px;
  background: #faf8f5;
  border: 1px solid #aaa;
  border-radius: 2px;
  cursor: ns-resize;
  pointer-events: auto;
  box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}
.step-slider-wrapper input[type="range"]::-webkit-slider-runnable-track {
  width: 100%;
  height: 100%;
  background: transparent;
  border: none;
  cursor: pointer;
}
.step-slider-wrapper input[type="range"]::-moz-range-track {
  width: 2px;
  height: 100%;
  background: transparent;
  border: none;
  cursor: pointer;
}
.step-slider-wrapper input[type="range"]::-ms-track {
  width: 100%;
  height: 100%;
  background: transparent;
  border-color: transparent;
  color: transparent;
}
.step-slider-wrapper input[type="range"]::-ms-fill-lower,
.step-slider-wrapper input[type="range"]::-ms-fill-upper {
  background: transparent;
}
</style>
"""

_DUAL_SLIDER_JS = """
<script>
(function() {
    var plotEl = document.querySelector('.plotly-graph-div');
    if (!plotEl) return;

    var steps = STEPS_ARRAY;
    var colorscaleName = 'COLORSCALE_NAME';
    var numTraces = steps.length;
    var stepMin = STEP_MIN;
    var stepMax = STEP_MAX;

    var colorscales = {
        'Archival': [[0,'#2e3a5f'],[0.25,'#2d6a7a'],[0.5,'#3a9a8c'],[0.75,'#7dbf7d'],[1,'#d4a84b']],
        'Turbo': [[0,'#30123b'],[0.1,'#4662d7'],[0.2,'#36aaf9'],[0.3,'#1ae4b6'],[0.4,'#72fe5e'],[0.5,'#c8ef34'],[0.6,'#faba39'],[0.7,'#f66b19'],[0.8,'#cb2a04'],[0.9,'#7a0403'],[1,'#7a0403']],
        'Viridis': [[0,'#440154'],[0.25,'#3b528b'],[0.5,'#21918c'],[0.75,'#5ec962'],[1,'#fde725']],
        'Plasma': [[0,'#0d0887'],[0.25,'#7e03a8'],[0.5,'#cc4778'],[0.75,'#f89540'],[1,'#f0f921']],
        'Inferno': [[0,'#000004'],[0.25,'#57106e'],[0.5,'#bc3754'],[0.75,'#f98e09'],[1,'#fcffa4']],
        'Cividis': [[0,'#00224e'],[0.25,'#49516f'],[0.5,'#8c8c8c'],[0.75,'#c5b552'],[1,'#fdea45']]
    };

    function interpolateColor(cs, t) {
        t = Math.max(0, Math.min(1, t));
        for (var i = 0; i < cs.length - 1; i++) {
            if (t >= cs[i][0] && t <= cs[i+1][0]) {
                var ratio = (t - cs[i][0]) / (cs[i+1][0] - cs[i][0]);
                var c1 = cs[i][1], c2 = cs[i+1][1];
                var r1 = parseInt(c1.slice(1,3), 16), g1 = parseInt(c1.slice(3,5), 16), b1 = parseInt(c1.slice(5,7), 16);
                var r2 = parseInt(c2.slice(1,3), 16), g2 = parseInt(c2.slice(3,5), 16), b2 = parseInt(c2.slice(5,7), 16);
                return 'rgb(' + Math.round(r1 + ratio*(r2-r1)) + ',' + Math.round(g1 + ratio*(g2-g1)) + ',' + Math.round(b1 + ratio*(b2-b1)) + ')';
            }
        }
        return cs[cs.length-1][1];
    }

    function updatePlot(minVal, maxVal) {
        var cs = colorscales[colorscaleName] || colorscales['Archival'];
        var visible = [];
        var colors = [];
        var denom = (maxVal - minVal) || 1;

        for (var i = 0; i < numTraces; i++) {
            var s = steps[i];
            if (s >= minVal && s <= maxVal) {
                visible.push(true);
                colors.push(interpolateColor(cs, (s - minVal) / denom));
            } else {
                visible.push(false);
                colors.push(null);
            }
        }

        Plotly.restyle(plotEl, {visible: visible, 'line.color': colors});
    }

    // Create vertical slider UI
    var container = document.createElement('div');
    container.className = 'step-slider-container';
    container.innerHTML =
        '<div class="step-slider-max-label" id="maxLabel">' + stepMax + '</div>' +
        '<div class="step-slider-wrapper">' +
        '  <div class="step-slider-track"></div>' +
        '  <div class="step-slider-range" id="sliderRange"></div>' +
        '  <input type="range" id="sliderMin" min="0" max="100" value="0" orient="vertical">' +
        '  <input type="range" id="sliderMax" min="0" max="100" value="100" orient="vertical">' +
        '</div>' +
        '<div class="step-slider-min-label" id="minLabel">' + stepMin + '</div>' +
        '<div class="step-slider-title">Training Step</div>';

    plotEl.style.position = 'relative';
    plotEl.appendChild(container);

    var sliderMin = document.getElementById('sliderMin');
    var sliderMax = document.getElementById('sliderMax');
    var rangeEl = document.getElementById('sliderRange');
    var minLabel = document.getElementById('minLabel');
    var maxLabel = document.getElementById('maxLabel');

    function pctToStep(pct) {
        return Math.round(stepMin + (pct / 100) * (stepMax - stepMin));
    }

    function updateSliderUI() {
        var minPct = Math.min(parseInt(sliderMin.value), parseInt(sliderMax.value));
        var maxPct = Math.max(parseInt(sliderMin.value), parseInt(sliderMax.value));
        var minStep = pctToStep(minPct);
        var maxStep = pctToStep(maxPct);

        rangeEl.style.bottom = minPct + '%';
        rangeEl.style.top = (100 - maxPct) + '%';
        minLabel.textContent = minStep;
        maxLabel.textContent = maxStep;
        updatePlot(minStep, maxStep);
    }

    sliderMin.addEventListener('input', updateSliderUI);
    sliderMax.addEventListener('input', updateSliderUI);

    // Initialize on load
    updateSliderUI();
})();
</script>
"""


def plot_cev_over_training(
    history: CEVHistory,
    layer_name: str,
    *,
    config: CEVVizConfig | Mapping[str, Any] | DictConfig | None = None,
    colorscale: str | None = None,
    history_window: int | None = None,
    max_components: int | None = None,
    width: int | None = None,
    height: int | None = None,
    title: str | None = None,
    title_suffix: str = "",
    show_rangeslider: bool | None = None,
    line_width: float | None = None,
    line_opacity: float | None = None,
    step_range: tuple[int, int] | None = None,
    y_range: tuple[float, float] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colorbar_title: str | None = None,
    colorbar_thickness: int | None = None,
    colorbar_len: float | None = None,
    show_step_slider: bool = True,
) -> go.Figure:
    """Plot CEV curves over training with an interactive step slider.

    The step slider allows filtering which training steps to display.
    Colors are dynamically recomputed based on the visible range.
    """
    cfg = _coerce_cev_config(config)

    colorscale = colorscale if colorscale is not None else cfg.colorscale
    history_window = history_window if history_window is not None else cfg.history_window
    max_components = max_components if max_components is not None else cfg.max_components
    width = width if width is not None else cfg.width
    height = height if height is not None else cfg.height
    show_rangeslider = show_rangeslider if show_rangeslider is not None else cfg.show_rangeslider
    line_width = line_width if line_width is not None else cfg.line_width
    line_opacity = line_opacity if line_opacity is not None else cfg.line_opacity
    y_range = y_range if y_range is not None else cfg.y_range
    x_label = x_label if x_label is not None else cfg.x_label
    y_label = y_label if y_label is not None else cfg.y_label
    colorbar_title = colorbar_title if colorbar_title is not None else cfg.colorbar_title
    colorbar_thickness = colorbar_thickness if colorbar_thickness is not None else int(COLORBAR_DEFAULTS["thickness"])
    colorbar_len = colorbar_len if colorbar_len is not None else float(COLORBAR_DEFAULTS["len"])

    if title is None:
        title = f"CEV Over Training - {layer_name}"
        if title_suffix:
            title = f"{title}<br><sup>{title_suffix}</sup>"

    if layer_name not in history:
        raise KeyError(f"Layer '{layer_name}' not found in CEV history (available: {list(history.keys())})")

    entries = list(history.get(layer_name) or [])
    if not entries:
        warnings.warn(f"CEV history for layer '{layer_name}' is empty.", UserWarning, stacklevel=2)
        fig = _empty_figure(title, "No CEV history available for this layer.")
        return apply_default_layout(fig, width=width, height=height, title=dict(text=title, font=TITLE_FONT))

    cleaned: list[tuple[int, np.ndarray]] = []
    for entry in entries:
        step = entry.get("step")
        cumvar = entry.get("cumvar")
        if not isinstance(step, int):
            warnings.warn(f"Skipping CEV entry with non-int step for layer '{layer_name}': {step!r}", UserWarning)
            continue
        if cumvar is None:
            warnings.warn(f"Skipping CEV entry missing 'cumvar' for layer '{layer_name}' at step {step}.", UserWarning)
            continue
        arr = np.asarray(cumvar, dtype=float).reshape(-1)
        if arr.size == 0:
            warnings.warn(f"Skipping empty cumvar array for layer '{layer_name}' at step {step}.", UserWarning)
            continue
        if not np.all(np.isfinite(arr)):
            warnings.warn(f"Skipping cumvar with NaN/Inf for layer '{layer_name}' at step {step}.", UserWarning)
            continue
        if max_components is not None:
            arr = arr[: int(max_components)]
        cleaned.append((step, arr))

    if not cleaned:
        warnings.warn(f"No valid CEV entries for layer '{layer_name}'.", UserWarning, stacklevel=2)
        fig = _empty_figure(title, "No valid CEV entries after filtering.")
        return apply_default_layout(fig, width=width, height=height, title=dict(text=title, font=TITLE_FONT))

    cleaned.sort(key=lambda t: t[0])
    if step_range is not None:
        min_step, max_step = step_range
        cleaned = [(s, a) for (s, a) in cleaned if min_step <= s <= max_step]

    if history_window is not None and history_window > 0:
        cleaned = cleaned[-int(history_window) :]

    if not cleaned:
        fig = _empty_figure(title, "No CEV entries in the requested step range.")
        return apply_default_layout(fig, width=width, height=height, title=dict(text=title, font=TITLE_FONT))

    steps = [s for (s, _) in cleaned]
    step_min = min(steps)
    step_max = max(steps)
    num_traces = len(cleaned)

    # For the default view (all steps), compute colors
    denom = (step_max - step_min) if step_max > step_min else 1
    norms = [(s - step_min) / denom for s in steps]
    # Use custom Archival colorscale if specified, otherwise use Plotly built-in
    cs = ARCHIVAL_COLORSCALE if colorscale == "Archival" else colorscale
    line_colors = plotly.colors.sample_colorscale(cs, norms)

    fig = go.Figure()

    # Add data traces
    for (step, cumvar), color in zip(cleaned, line_colors, strict=False):
        x = np.arange(1, cumvar.shape[0] + 1, dtype=int)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=cumvar,
                mode="lines",
                line=dict(color=color, width=float(line_width)),
                opacity=float(line_opacity),
                hovertemplate="Step: %{customdata}<br>Component: %{x}<br>CEV: %{y:.4f}<extra></extra>",
                customdata=np.full_like(x, step, dtype=int),
                showlegend=False,
            )
        )

    # Note: Colorbar removed - replaced by interactive vertical slider in HTML

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label, range=list(y_range))
    if show_rangeslider:
        fig.update_xaxes(rangeslider=dict(visible=True))

    # Store metadata for dual slider injection in write_cev_html
    if show_step_slider and num_traces > 1:
        step_interval = steps[1] - steps[0] if len(steps) > 1 else 1000
        fig._cev_slider_meta = {
            "steps": steps,
            "colorscale": colorscale,
            "step_min": step_min,
            "step_max": step_max,
            "step_interval": step_interval,
        }

    return apply_default_layout(fig, width=width, height=height, title=dict(text=title, font=TITLE_FONT))


def write_cev_html(fig: go.Figure, path: str, include_plotlyjs: str = "cdn") -> None:
    """Write CEV figure to HTML with dual-handle step range slider."""
    html_content = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)

    # Inject dual slider if metadata exists
    meta = getattr(fig, "_cev_slider_meta", None)
    if meta:
        css = _DUAL_SLIDER_CSS
        js = _DUAL_SLIDER_JS
        js = js.replace("STEPS_ARRAY", str(meta["steps"]))
        js = js.replace("COLORSCALE_NAME", meta["colorscale"])
        js = js.replace("STEP_MIN", str(meta["step_min"]))
        js = js.replace("STEP_MAX", str(meta["step_max"]))
        js = js.replace("STEP_INTERVAL", str(meta["step_interval"]))
        # Insert CSS in head, JS before </body>
        html_content = html_content.replace("</head>", css + "</head>")
        html_content = html_content.replace("</body>", js + "</body>")

    with open(path, "w") as f:
        f.write(html_content)


# Matches keys like "pca/cev/L0.resid.pre" (format: analysis/cev/layer)
_CEV_ARRAY_KEY_RE = re.compile(r"^(?P<analysis>[^/]+)/cev/(?P<layer>.+)$")


def update_cev_history(
    history: CEVHistory,
    arrays: Mapping[str, Any],
    *,
    step: int,
    analysis_name: str = "pca",
    max_components: int | None = None,
) -> None:
    """Update a CEV history dict from array outputs (e.g., 'pca/cev/L0.resid.pre').

    The PCA analysis outputs cumulative explained variance as an array with key
    '{analysis_name}/cev/{layer}'. This function extracts those arrays and adds
    them to the history.
    """
    prefix = f"{analysis_name}/"
    for key, value in arrays.items():
        if not key.startswith(prefix):
            continue
        match = _CEV_ARRAY_KEY_RE.match(key)
        if not match or match.group("analysis") != analysis_name:
            continue

        layer = match.group("layer")
        cumvar = np.asarray(value, dtype=float).reshape(-1)

        if cumvar.size == 0:
            continue
        if not np.all(np.isfinite(cumvar)):
            warnings.warn(f"Skipping CEV array with NaN/Inf for layer '{layer}' at step {step}.")
            continue

        if max_components is not None:
            cumvar = cumvar[: int(max_components)]

        history.setdefault(layer, []).append({"step": int(step), "cumvar": cumvar})


__all__ = ["plot_cev_over_training", "update_cev_history", "write_cev_html"]
