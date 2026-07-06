# RSMA Simulator

A web-based interactive simulator for comparing Rate-Splitting Multiple Access (RSMA) against Non-Orthogonal Multiple Access (NOMA) and Orthogonal Multiple Access (OMA) in MISO downlink systems.

## Live Demo

🚀 **Try the simulator:** [https://rsmasimulator.vercel.app/](https://rsmasimulator.vercel.app/)

## Features

- **Interactive Parameter Control**: Adjust system parameters in real-time including:
  - Number of base station antennas (Nt)
  - Channel gains for near and far users
  - Spatial angle between users
  - SNR and noise power
  - CSI estimation error
  - Power allocation ratios

- **Automatic Optimization**: Grid search optimization for RSMA power allocation with QoS constraints

- **Real-time Visualization**: 
  - Sum rate comparison across RSMA, NOMA, and OMA
  - Performance vs SNR graphs
  - Individual user rate breakdowns

- **Mathematical Rigor**: Implements proper:
  - ULA steering vectors for spatial correlation
  - Zero-Forcing (ZF) beamforming for private streams
  - Maximum Ratio Transmission (MRT) for common streams
  - Interference-aware rate calculations
  - Successive Interference Cancellation (SIC) for NOMA

## Technical Details

### Channel Model
- Uniform Linear Array (ULA) with steering vectors: `h = [1, e^(jπ sin(θ)), ...]`
- Spatial correlation based on angular separation
- Optional CSI estimation error modeling

### Rate Calculations

**RSMA:**
- Common rate: `Rc = min(Rc1, Rc2)` where both users decode common stream first
- Private rates include interference from other user's private stream
- Common rate split proportionally based on channel strengths

**NOMA:**
- User 2 (far): Treats User 1 as noise
- User 1 (near): Uses SIC to remove User 2's signal

**OMA:**
- Time-division multiple access with beamforming
- Each user gets 50% time slot with dedicated beamformer

### Beamforming
- ZF precoders for private streams (minimize inter-user interference)
- MRT precoder for common stream (maximize coverage)

## Getting Started

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/Tejas-200/RSMA-Simulator.git
cd RSMA-Simulator

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

### Deployment

The app is deployed on Vercel. To deploy your own version:

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

## Usage

1. **Set System Parameters**: Use the sliders to configure antenna count, channel gains, spatial angle, SNR, etc.

2. **Choose Power Allocation**: 
   - Manual: Adjust RSMA power ratios directly
   - Auto-optimize: Enable automatic grid search for optimal power allocation

3. **View Results**: 
   - Current performance at selected SNR
   - Sum rate comparison graph across SNR range
   - Individual user rates and gains

4. **Experiment**: Try different scenarios:
   - High channel gap (strong disparity between users)
   - Small spatial angles (high correlation)
   - Imperfect CSI
   - Different antenna configurations

## Key Insights

- **RSMA Flexibility**: RSMA can replicate OMA (Pc=0) and NOMA (P2=0) by adjusting power allocation
- **Spatial Correlation**: RSMA handles spatial correlation better than NOMA
- **QoS Support**: Optimizer respects minimum rate constraints for both users
- **Adaptive Performance**: RSMA naturally adapts to channel conditions

## Technologies

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Styling
- **Lucide React**: Icons
- **Recharts**: Data visualization (optional, currently using Canvas API)

## Mathematical Framework

The simulator implements the standard RSMA rate equations:

```
Rc1 = log2(1 + |h1^H wc|^2 Pc / (|h1^H w1|^2 P1 + |h1^H w2|^2 P2 + σ^2))
Rc2 = log2(1 + |h2^H wc|^2 Pc / (|h2^H w1|^2 P1 + |h2^H w2|^2 P2 + σ^2))
Rc = min(Rc1, Rc2)

R1_RSMA = C1 + log2(1 + |h1^H w1|^2 P1 / (|h1^H w2|^2 P2 + σ^2))
R2_RSMA = C2 + log2(1 + |h2^H w2|^2 P2 / (|h2^H w1|^2 P1 + σ^2))
```

Where C1 and C2 are the common rate portions allocated to each user based on channel strengths.

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This simulator implements the Rate-Splitting Multiple Access framework as described in wireless communications research literature.
