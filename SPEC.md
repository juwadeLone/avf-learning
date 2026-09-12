# saca-AVF reproduction — design spec

Paper: Tan et al., "Saca-AVF: A Quantitative Approach to Analyze the Architectural
Vulnerability Factors of CNN Accelerators", IEEE TC 72(11), 2023.
Text dump of the paper: /tmp/paper.txt (sections III–V are the relevant ones).

Goal: compute saca-AVF (eq. 3) of a systolic-array CNN accelerator for
{WS, IS, OS} x {32,64,128,256,512}^2 arrays x {LeNet-5, Cifar-10 CNN, VGG-16},
reproduce Fig. 6 (model-level AVF), Fig. 12 (layer-wise AVF), Fig. 11 (per-PE
AVF heatmaps) and Fig. 7 (PE utilization).

## 1. Definitions

Each PE has three 32-bit registers: `ifmap`, `weight`, `psum` (Table II: fp32).
`BITS = 32`.

saca-AVF of a run = (sum over cycles, PEs, registers of ACE bits)
                  / (3 * BITS * PEs_total * cycles_total)              (eq. 3)

`PEs_total = arr_h * arr_w` always (idle PEs count in the denominator).
`cycles_total` = sum over all layers and all tiles of the tile's cycle count.
Model-level AVF is the *single ratio* over the whole network, not the mean of
layer ratios. Layer-level AVF = same ratio restricted to that layer's cycles.
Per-PE AVF (eq. 2) = that PE's ACE bits / (3 * BITS * cycles of the layer).
PE utilization (Fig. 7) = (sum over tiles of active_PEs * tile_cycles)
                        / (PEs_total * cycles_total), where active PEs of a tile
are the PEs inside the tile's (rows x cols) footprint.

## 2. ACE rules (paper §III-A, Table I)

A register bit at a given cycle is ACE iff the value currently held in it will
still be consumed by a computation that reaches the output, AND that
consumption is not directly masked by a multiplication with zero:

| ifmap | weight | ACE bits (this MAC)              |
|-------|--------|----------------------------------|
| !=0   | !=0    | ifmap, weight, psum              |
| ==0   | !=0    | ifmap, psum   (weight masked)    |
| !=0   | ==0    | weight, psum  (ifmap masked)     |
| ==0   | ==0    | psum only                        |

- `psum` is ACE whenever the PE is holding a live partial sum (from the cycle of
  its first MAC in a tile until the cycle the value leaves the PE). A flip in a
  psum always propagates additively; it is never masked.
- A stationary operand (weight in WS, ifmap in IS) is ACE
  * during pre-store cycles once it has entered the array (paper Fig. 4:
    3, 6, 9 registers for a 3x3 array in cycles 1,2,3) — treat every pre-stored
    value as ACE while in flight / resident before its first use;
  * during MAC cycles when it is being multiplied by a non-zero operand
    (Table I);
  * during MAC cycles when it is resident but the PE is waiting for its first
    streamed operand (skew bubbles at the start): ACE (it will still be used);
  * after its last use in the tile (dynamically dead): un-ACE.
- Streamed registers (ifmap/psum in WS, weight/psum in IS, ifmap/weight in OS)
  are un-ACE when the PE is idle in that cycle (skew bubbles, pre-store cycles).
- PEs outside the tile footprint: all three registers un-ACE.

## 3. Layer -> matrices

For every CONV / FC layer use im2col:
- `I` : N x K   (N = output pixels = OH*OW, K = R*S*C)  — ifmap matrix
- `W` : K x M   (M = number of filters)                  — filter matrix
- `O = I @ W` : N x M
For FC layers: N = 1, K = in_features, M = out_features.
Values come from real inference (Keras) on real test images, layer by layer
(ifmap of layer l = actual activations after ReLU/pooling of layer l-1).
Bias, ReLU, pooling, softmax are NOT executed on the array (paper §V-D).

Zero masks: `Iz = (I == 0)`, `Wz = (W == 0)` (exact zero; fp32 after ReLU is
exactly 0.0 — do not use a tolerance).

## 4. Dataflow timing model (SCALE-SIM-style, one PE hop per cycle)

Let `ah = arr_h`, `aw = arr_w`. All tile cycle counts and ACE counts below are
closed-form so that big layers (VGG-16) are cheap: cost is O(N*K + K*M) numpy
per tile, never a per-cycle loop over the whole run.

### 4.1 Weight Stationary (WS)
Tiling: K split into row-tiles of height <= ah, M split into col-tiles of
width <= aw. Loop order: for each M-tile, for each K-tile, stream all N rows of
I. Tile footprint: kh rows x mw cols. PE(i,j) holds W[k_i, m_j].
- Pre-store: `kh` cycles. In pre-store cycle c (c = 1..kh) rows 0..c-1 hold a
  (transiting or final) weight -> weight reg ACE for c*mw registers. ifmap/psum
  un-ACE.
- MAC phase: I row n (n = 0..N-1) reaches PE(i,j) at MAC-cycle t = n + i + j.
  Phase length = N + kh + mw - 2 cycles. Tile cycles = kh + N + kh + mw - 2.
- Per PE(i,j) over the MAC phase (nz = "non-zero"):
  * weight ACE cycles = (i + j)   [waiting for first ifmap, still live]
                      + count_n( I[n,k_i] != 0 )
  * ifmap  ACE cycles = count_n( W[k_i,m_j] != 0 ) = N if W[k_i,m_j]!=0 else 0
  * psum   ACE cycles = N
  (after n = N-1 passes, all three are dead/idle -> un-ACE)
  Vectorised: `nzI_col[k] = (I[:,k] != 0).sum()` gives weight-ACE per row i for
  all j at once.

### 4.2 Input Stationary (IS)
Mirror of WS with roles of I and W swapped. Stationary operand is `I^T`
(K x N): tile kh rows of K x nw cols of N; PE(i,j) holds I[n_j, k_i].
Streamed operand: rows of `W^T` (M rows, each of length K), m = 0..M-1
reaching PE(i,j) at t = m + i + j. Loop: for each N-tile, for each K-tile,
stream all M rows.
- Pre-store: kh cycles, ifmap regs ACE (c*nw registers in cycle c).
- MAC phase length M + kh + nw - 2. Tile cycles = kh + M + kh + nw - 2.
- Per PE(i,j): ifmap ACE = (i+j) + count_m( W[k_i,m] != 0 );
  weight ACE = M if I[n_j,k_i] != 0 else 0; psum ACE = M.

### 4.3 Output Stationary (OS)
Tile: nh rows of N x mw cols of M; PE(i,j) accumulates O[n_i, m_j] over all
K. No pre-store. I row n_i streams from the left along row i (element k at
cycle k + i + j), W column m_j streams from the top along column j (same
timing). Loop: for each N-tile, for each M-tile, stream all K.
- Tile cycles = K + nh + mw - 2 (last PE finishes at cycle K-1 + nh-1 + mw-1;
  the result is read out in that cycle).
- Per PE(i,j): first MAC at cycle i+j, last at K-1+i+j.
  * ifmap  ACE cycles = count_k( W[k,m_j] != 0 )
  * weight ACE cycles = count_k( I[n_i,k] != 0 )
  * psum   ACE cycles = (cycles from first MAC to end of tile)
                      = (K + nh + mw - 2) - (i + j)
  Vectorised: `nzW_col[m] = (W[:,m]!=0).sum()`, `nzI_row[n] = (I[n,:]!=0).sum()`.

## 5. Two implementations, cross-checked

1. `analytic.py` — the closed-form counts above (used for all real experiments).
2. `cyclesim.py` — a literal cycle-by-cycle simulator of the same timing model:
   keeps a (ah, aw, 3) register state and an ACE mask per cycle, moves data one
   hop per cycle, applies the Table I rules per PE per cycle. Slow; only for
   validation on tiny inputs.
Test: for random small I, W (with ~50% zeros, incl. all-zero rows/cols) and
arrays 2x2..5x5, both implementations must give identical total ACE-bit
counts, identical cycle counts, and `cyclesim` must also reproduce `O = I@W`
exactly (proves the dataflow actually computes the right thing).

## 6. Models and data (Table III)

- LeNet-5 on MNIST. The paper's LeNet-5 is a variant (Fig. 11 text: layer-1
  filter is 3x3x32; "7th layer" is FC). Use this Keras model, ReLU everywhere,
  softmax at the end, train 2 epochs on CPU:
  Conv2D(32,3)-Conv2D(64,3)-MaxPool-Conv2D(64,3)-MaxPool-Flatten-Dense(128)-
  Dense(84)-Dense(10). MAC layers = 6. Document the deviation in the report.
- Cifar-10 CNN: the Keras cifar10_cnn example:
  Conv(32,3)-Conv(32,3)-Pool-Conv(64,3)-Conv(64,3)-Pool-Flatten-Dense(512)-
  Dense(10), ReLU, train ~10 epochs (aim ~70% acc; accuracy is not important,
  zero-distribution is). MAC layers = 6.
- VGG-16: `keras.applications.VGG16(weights="imagenet")`. Inputs: 100 natural
  images at 224x224. If ImageNet val images are not obtainable, use 100 images
  from a freely downloadable natural-image set (e.g. `imagenette` via
  tensorflow_datasets, or torchvision-free download) with VGG preprocessing.
  Document which.
- Sample 100 test images per model (fixed seed 0). For VGG-16, if runtime is a
  problem, use fewer (>=20) and report N used and std.

## 7. Experiments / outputs (write to `results/`)

- `fig6_model_avf.csv`: model, dataflow, array, mean AVF, std over images.
- `fig7_pe_util.csv`: same keys, PE utilization.
- `fig12_layer_avf.csv`: model, dataflow, array, layer, AVF (mean over images).
- `fig11_pe_maps.npz`: per-PE AVF arrays for (LeNet-5 L1, WS, 32x32),
  (Cifar-10 L4, IS, 32x32), (VGG-16 L1, OS, 64x64), first image.
- Plots (matplotlib, png): fig6.png, fig7.png, fig12.png, fig11.png.
- Sanity checks against the paper (Cifar-10 OS): 32->34.8%, 64->20.9%,
  128->8.7%, 256->3.3%, 512->1.0%; and OS > IS > WS for a given size.
  Report our numbers side by side; do not tune anything to match.
