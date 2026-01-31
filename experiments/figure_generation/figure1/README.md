# Figure 1: Factored Belief Geometry

Scripts and outputs for generating Figure 1 of the paper, which visualizes the factored world hypothesis using Hidden Markov Models and geometric representations.

## Folder Structure

```
figure1/
├── scripts/          # All Python scripts
├── pdf/              # PDF outputs
├── png/              # PNG outputs
├── svg/              # SVG outputs
└── README.md
```

## Scripts

### Matplotlib-based (run with Python)

| Script | Description | Outputs |
|--------|-------------|---------|
| `segre_surface.py` | Main matplotlib figure: 3D Segre surface + 2D factored squares for independent, dependent, and indecomposable beliefs | `segre_geometry.*` |
| `hmm_diagrams.py` | Joint 4-state HMM + Factored 2x2 HMM diagrams with transition probabilities | `hmm_diagrams.*` |
| `paper_figure.py` | Combined ICML paper figure (HMMs + Segre + scaling plot) | `paper_figure.*` |
| `factored_square.py` | Standalone 2D factored square view for indecomposable beliefs | `factored_square.{png,pdf}` |
| `figure_scaling.py` | Scaling plots showing dimensionality comparison | `scaling_{log,linear,both,log_fitted}.*` |
| `mess3_beliefs.py` | Combined 3-factor belief visualization (requires `uv run`) | `mess3_beliefs.*` |

### Blender-based (run with Blender)

| Script | Description | Outputs |
|--------|-------------|---------|
| `blender_test.py` | Minimal test script (red sphere on white background) | `blender_test.png` |
| `blender_lowres.py` | Fast preview (8 samples, 400x350) for indecomposable case | `blender_lowres_{3d,2d}.png` |
| `blender_single.py` | High-quality render for indecomposable case (64 samples) | `blender_indecomposable_{3d,2d}.png` |
| `blender_all.py` | All 3 cases (independent, dependent, indecomposable) in 3D and 2D | `blender_{independent,dependent,indecomposable}_{3d,2d}.png` |
| `freeze_vary.py` | Freeze-and-vary visualization showing parallel lines in factored space | `freeze_vary_{3d,2d,2d_centered,3d_centered}.png` |
| `mess3_blender.py` | 3-factor belief visualization showing factor simplexes and joint space | `mess3_{factor_simplexes,joint_beliefs}.png` |

## Running the Scripts

### Matplotlib scripts

```bash
cd /path/to/simplex-research
python experiments/figure_generation/figure1/scripts/segre_surface.py
python experiments/figure_generation/figure1/scripts/hmm_diagrams.py
python experiments/figure_generation/figure1/scripts/paper_figure.py
python experiments/figure_generation/figure1/scripts/factored_square.py
python experiments/figure_generation/figure1/scripts/figure_scaling.py

# Use uv to manage dependencies
uv run python experiments/figure_generation/figure1/scripts/mess3_beliefs.py
```

### Blender scripts

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_test.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_lowres.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_single.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_all.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/freeze_vary.py
/Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/mess3_blender.py
```

## Key Concepts

- **Segre Surface**: The image of the Segre embedding, representing factored beliefs in 3D space
- **Factored Square**: 2D representation where each axis corresponds to a binary factor's belief state
- **Independent**: Both factors evolve independently (stays on Segre surface)
- **Dependent**: Factors are coupled but still decomposable (stays on Segre surface)
- **Indecomposable**: True joint beliefs that cannot be factored (OFF the Segre surface)
- **Freeze-and-Vary**: Demonstrates that varying one factor while freezing another creates parallel lines in factored space but curved paths with different tangents in joint space
