import React, { useState, useEffect } from 'react';

function Slider({ label, value, onChange, min, max, step, unit = '' }) {
  return (
    <div style={{ marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <label style={{ fontSize: '14px', fontWeight: '500', color: '#374151' }}>{label}</label>
        <span style={{ fontSize: '14px', fontFamily: 'monospace', backgroundColor: '#dbeafe', color: '#1e40af', padding: '2px 8px', borderRadius: '4px' }}>
          {value.toFixed(step < 1 ? 2 : 1)}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', height: '8px', borderRadius: '4px', cursor: 'pointer', accentColor: '#2563eb' }}
      />
    </div>
  );
}

// Helper: Generate complex channel vector with spatial correlation
function generateChannelVector(Nt, magnitude, angleDeg, csiError = 0.0) {
  const angleRad = angleDeg * Math.PI / 180;
  const vector = [];
  for (let i = 0; i < Nt; i++) {
    // Uniform linear array steering vector: phase = π * sin(angle) * i
    const phase = Math.PI * Math.sin(angleRad) * i;
    const real = magnitude * Math.cos(phase);
    const imag = magnitude * Math.sin(phase);
    
    // Add CSI estimation error if imperfect CSI
    if (csiError > 0) {
      const errorReal = (Math.random() - 0.5) * 2 * csiError * magnitude;
      const errorImag = (Math.random() - 0.5) * 2 * csiError * magnitude;
      vector.push({
        real: real + errorReal,
        imag: imag + errorImag
      });
    } else {
      vector.push({ real, imag });
    }
  }
  return vector;
}

// Helper: Complex vector dot product
function complexDotProduct(h, w) {
  let result = { real: 0, imag: 0 };
  for (let i = 0; i < h.length; i++) {
    result.real += h[i].real * w[i].real - h[i].imag * w[i].imag;
    result.imag += h[i].real * w[i].imag + h[i].imag * w[i].real;
  }
  return result;
}

// Helper: Complex magnitude squared
function complexMagSquared(z) {
  return z.real * z.real + z.imag * z.imag;
}

// Helper: Vector norm squared
function vectorNormSquared(v) {
  return v.reduce((sum, elem) => sum + elem.real * elem.real + elem.imag * elem.imag, 0);
}

// Helper: Normalize vector to given power
function normalizeVector(v, targetPower) {
  const currentNorm = Math.sqrt(vectorNormSquared(v));
  if (currentNorm === 0) return v;
  const scale = Math.sqrt(targetPower) / currentNorm;
  return v.map(elem => ({
    real: elem.real * scale,
    imag: elem.imag * scale
  }));
}

// Helper: Zero-Forcing precoder for private streams
function computeZFPrecoder(h1, h2, power) {
  const Nt = h1.length;
  
  // To make w1 orthogonal to h2, we project h1 onto the null space of h2
  // Formula: w1 = h1 - ( (h2^H * h1) / ||h2||^2 ) * h2
  
  // 1. Compute h2^H * h1 (complex inner product)
  let h2_proj_h1 = { real: 0, imag: 0 };
  for (let i = 0; i < Nt; i++) {
    // h2^H means conjugate of h2
    h2_proj_h1.real += h2[i].real * h1[i].real + h2[i].imag * h1[i].imag;
    h2_proj_h1.imag += h2[i].real * h1[i].imag - h2[i].imag * h1[i].real;
  }
  
  const h2NormSq = vectorNormSquared(h2);
  
  // 2. Compute w1 vector components
  const w1 = [];
  for (let i = 0; i < Nt; i++) {
    if (h2NormSq === 0) {
      w1.push({ ...h1[i] });
    } else {
      // Subtraction step: h1 - (h2_proj_h1 / h2NormSq) * h2
      const projReal = (h2_proj_h1.real * h2[i].real - h2_proj_h1.imag * h2[i].imag) / h2NormSq;
      const projImag = (h2_proj_h1.real * h2[i].imag + h2_proj_h1.imag * h2[i].real) / h2NormSq;
      w1.push({
        real: h1[i].real - projReal,
        imag: h1[i].imag - projImag
      });
    }
  }
  
  // 3. Repeat process for w2 (orthogonal to h1)
  let h1_proj_h2 = { real: 0, imag: 0 };
  for (let i = 0; i < Nt; i++) {
    h1_proj_h2.real += h1[i].real * h2[i].real + h1[i].imag * h2[i].imag;
    h1_proj_h2.imag += h1[i].real * h2[i].imag - h1[i].imag * h2[i].real;
  }
  const h1NormSq = vectorNormSquared(h1);
  
  const w2 = [];
  for (let i = 0; i < Nt; i++) {
    if (h1NormSq === 0) {
      w2.push({ ...h2[i] });
    } else {
      const projReal = (h1_proj_h2.real * h1[i].real - h1_proj_h2.imag * h1[i].imag) / h1NormSq;
      const projImag = (h1_proj_h2.real * h1[i].imag + h1_proj_h2.imag * h1[i].real) / h1NormSq;
      w2.push({
        real: h2[i].real - projReal,
        imag: h2[i].imag - projImag
      });
    }
  }
  
  return { 
    w1: normalizeVector(w1, power), 
    w2: normalizeVector(w2, power) 
  };
}

// Helper: MRT precoder for common stream (points toward both users)
function computeMRTPrecoder(h1, h2, power) {
  // MRT: w is proportional to h1 + h2
  const wc = h1.map((elem, i) => ({
    real: elem.real + h2[i].real,
    imag: elem.imag + h2[i].imag
  }));
  
  return normalizeVector(wc, power);
}

function runRSMASimulation(params) {
  const { Nt, h1, h2, spatialAngle, SNR, noisePower, csiError, commonPowerRatio, privatePower1Ratio, privatePower2Ratio } = params;
  
  // Convert SNR from dB to linear power
  const P = noisePower * Math.pow(10, SNR / 10);
  
  // Generate channel vectors with CSI error
  const h1_vec = generateChannelVector(Nt, h1, 0, csiError); // User 1 at 0 degrees
  const h2_vec = generateChannelVector(Nt, h2, spatialAngle, csiError); // User 2 at spatialAngle degrees
  
  const Pc = P * commonPowerRatio;
  const P1 = P * privatePower1Ratio;
  const P2 = P * privatePower2Ratio;
  
  // RSMA beamforming: ZF for private streams, MRT for common stream
  const { w1, w2 } = computeZFPrecoder(h1_vec, h2_vec, 1.0); 
  const wc = computeMRTPrecoder(h1_vec, h2_vec, 1.0);
  
  // Scale beamformers with allocated power
  const w1_scaled = normalizeVector(w1, P1);
  const w2_scaled = normalizeVector(w2, P2);
  const wc_scaled = normalizeVector(wc, Pc);
  
  // Compute effective complex channel gains
  const h1_wc = complexDotProduct(h1_vec, wc_scaled);
  const h1_w1 = complexDotProduct(h1_vec, w1_scaled);
  const h1_w2 = complexDotProduct(h1_vec, w2_scaled);
  
  const h2_wc = complexDotProduct(h2_vec, wc_scaled);
  const h2_w1 = complexDotProduct(h2_vec, w1_scaled);
  const h2_w2 = complexDotProduct(h2_vec, w2_scaled);
  
  // Common Rate calculation (both users decode common first)
  // Rc1 = log2(1 + |h1^H wc|^2 Pc / (|h1^H w1|^2 P1 + |h1^H w2|^2 P2 + σ^2))
  // Rc2 = log2(1 + |h2^H wc|^2 Pc / (|h2^H w1|^2 P1 + |h2^H w2|^2 P2 + σ^2))
  const Rc1 = Math.log2(1 + complexMagSquared(h1_wc) * Pc / (complexMagSquared(h1_w1) + complexMagSquared(h1_w2) + noisePower));
  const Rc2 = Math.log2(1 + complexMagSquared(h2_wc) * Pc / (complexMagSquared(h2_w1) + complexMagSquared(h2_w2) + noisePower));
  
  // Standard RSMA Rate tracking
  // Rc = min(Rc1, Rc2) = C1 + C2 (common stream rate split between users)
  const Rc = Math.min(Rc1, Rc2);
  
  // Private rates with interference from other user's private stream
  // R1^RSMA = C1 + log2(1 + |h1^H w1|^2 P1 / (|h1^H w2|^2 P2 + σ^2))
  // R2^RSMA = C2 + log2(1 + |h2^H w2|^2 P2 / (|h2^H w1|^2 P1 + σ^2))
  const Rp1 = Math.log2(1 + complexMagSquared(h1_w1) / (complexMagSquared(h1_w2) + noisePower));
  const Rp2 = Math.log2(1 + complexMagSquared(h2_w2) / (complexMagSquared(h2_w1) + noisePower));
  
  // Split common rate proportionally based on user channel strengths
  const h1_gain = complexMagSquared(h1_wc);
  const h2_gain = complexMagSquared(h2_wc);
  const total_gain = h1_gain + h2_gain;
  const C1 = total_gain > 0 ? Rc * (h1_gain / total_gain) : Rc / 2;
  const C2 = total_gain > 0 ? Rc * (h2_gain / total_gain) : Rc / 2;
  
  return {
    rate1: C1 + Rp1,
    rate2: C2 + Rp2,
    commonRate: Rc,
    sumRate: Rc + Rp1 + Rp2,
    technique: 'RSMA'
  };
}

// Grid search for optimal RSMA power allocation with QoS constraints
function optimizeRSMAPower(params) {
  let bestSumRate = 0;
  let bestRatios = { commonPowerRatio: 0.33, privatePower1Ratio: 0.34, privatePower2Ratio: 0.33 };
  let qosSatisfied = false;
  let initialized = false;
  
  // Fixed step intervals using integers to ensure perfect boundary tests
  for (let i = 0; i <= 100; i++) {
    const pc = i / 100;
    for (let j = 0; j <= 100 - i; j++) {
      const p1 = j / 100;
      const p2 = Math.max(0, 1.0 - pc - p1); 
      
      // Skip zero power allocations
      if (pc + p1 + p2 < 0.01) continue;
      
      const testParams = { ...params, commonPowerRatio: pc, privatePower1Ratio: p1, privatePower2Ratio: p2 };
      const result = runRSMASimulation(testParams);
      
      // QoS constraint: both users must achieve minimum rate
      if (result.rate1 >= params.minRate && result.rate2 >= params.minRate) {
        if (result.sumRate > bestSumRate || !qosSatisfied) {
          bestSumRate = result.sumRate;
          bestRatios = { commonPowerRatio: pc, privatePower1Ratio: p1, privatePower2Ratio: p2 };
          qosSatisfied = true;
        }
      } else if (!qosSatisfied) {
        // Fall back to unconstrained optimization if no QoS-satisfying allocation found
        if (!initialized || result.sumRate > bestSumRate) {
          bestSumRate = result.sumRate;
          bestRatios = { commonPowerRatio: pc, privatePower1Ratio: p1, privatePower2Ratio: p2 };
          initialized = true;
        }
      }
    }
  }
  return { bestSumRate, bestRatios };
}


function runNOMASimulation(params) {
  const { Nt, h1, h2, spatialAngle, SNR, noisePower, csiError, nomaPowerRatio } = params;
  
  // Convert SNR from dB to linear power
  const P = noisePower * Math.pow(10, SNR / 10);
  
  // Generate channel vectors with CSI error
  const h1_vec = generateChannelVector(Nt, h1, 0, csiError);
  const h2_vec = generateChannelVector(Nt, h2, spatialAngle, csiError);
  
  // NOMA in MISO: Use separate beamformers for each user (similar to RSMA private streams)
  // ZF beamformers to minimize inter-user interference
  const { w1, w2 } = computeZFPrecoder(h1_vec, h2_vec, 1.0);
  
  // Power allocation
  const a1 = 1 - nomaPowerRatio; // Power for strong user
  const a2 = nomaPowerRatio; // Power for weak user
  const P1 = P * a1;
  const P2 = P * a2;
  
  // Scale beamformers with allocated power
  const w1_scaled = normalizeVector(w1, P1);
  const w2_scaled = normalizeVector(w2, P2);
  
  // Effective channel gains with interference
  const h1_w1 = complexDotProduct(h1_vec, w1_scaled);
  const h1_w2 = complexDotProduct(h1_vec, w2_scaled);
  const h2_w1 = complexDotProduct(h2_vec, w1_scaled);
  const h2_w2 = complexDotProduct(h2_vec, w2_scaled);
  
  // User 2 (Far) Rate: Treats User 1 as noise
  const R2 = Math.log2(1 + complexMagSquared(h2_w2) / (complexMagSquared(h2_w1) + noisePower));
  
  // User 1 (Near) Rate: Uses SIC to remove User 2's signal (perfect SIC assumed)
  const R1 = Math.log2(1 + complexMagSquared(h1_w1) / noisePower);
  
  const sumRate = R1 + R2;
  
  return {
    rate1: R1,
    rate2: R2,
    sumRate,
    technique: 'NOMA'
  };
}

function runOMASimulation(params) {
  const { Nt, h1, h2, spatialAngle, SNR, noisePower, csiError } = params;
  
  // Convert SNR from dB to linear power
  const P = noisePower * Math.pow(10, SNR / 10);
  
  // Generate channel vectors with CSI error
  const h1_vec = generateChannelVector(Nt, h1, 0, csiError);
  const h2_vec = generateChannelVector(Nt, h2, spatialAngle, csiError);
  
  // OMA in MISO: TDMA with beamforming
  // Each user gets half the time with their own MRT beamformer
  
  // User 1 beamformer (MRT toward h1)
  const w1 = normalizeVector(h1_vec, P);
  const h1_w1 = complexDotProduct(h1_vec, w1);
  
  // User 2 beamformer (MRT toward h2)
  const w2 = normalizeVector(h2_vec, P);
  const h2_w2 = complexDotProduct(h2_vec, w2);
  
  // OMA: 50/50 time division
  const R1 = 0.5 * Math.log2(1 + complexMagSquared(h1_w1) / noisePower);
  const R2 = 0.5 * Math.log2(1 + complexMagSquared(h2_w2) / noisePower);
  
  const sumRate = R1 + R2;
  
  return {
    rate1: R1,
    rate2: R2,
    sumRate,
    technique: 'OMA'
  };
}

function App() {
  const [params, setParams] = useState({
    Nt: 2, // Base station antennas
    h1: 1.0, // Near user (strong channel magnitude)
    h2: 0.1, // Far user (weak channel magnitude)
    spatialAngle: 30, // Spatial angle between users (degrees)
    SNR: 10.0, // SNR in dB
    noisePower: 1.0, // Noise power (σ²)
    csiError: 0.0, // CSI estimation error (0 = perfect CSI)
    graphYMax: 10.0, // Y-axis maximum for graph (auto-scale if 0)
    minRate: 0.5, // Minimum rate constraint for QoS (bps/Hz)
    commonPowerRatio: 0.3, // Pc/P for RSMA (more balanced)
    privatePower1Ratio: 0.5, // P1/P for RSMA
    privatePower2Ratio: 0.2, // P2/P for RSMA
    nomaPowerRatio: 0.8, // a2 for NOMA (power to weak user)
    autoOptimize: false, // Auto-optimize RSMA power ratios
  });
  
  const [results, setResults] = useState({
    rsma: null,
    noma: null,
    oma: null,
  });
  
  const [optimizedRatios, setOptimizedRatios] = useState(null);
  
  useEffect(() => {
    if (params.autoOptimize) {
      // Auto-optimize RSMA power ratios
      const { bestRatios } = optimizeRSMAPower(params);
      setOptimizedRatios(bestRatios);
      const optimizedParams = { ...params, ...bestRatios };
      setResults({
        rsma: runRSMASimulation(optimizedParams),
        noma: runNOMASimulation(params),
        oma: runOMASimulation(params),
      });
    } else {
      setOptimizedRatios(null);
      setResults({
        rsma: runRSMASimulation(params),
        noma: runNOMASimulation(params),
        oma: runOMASimulation(params),
      });
    }
  }, [params]);
  
  const updateParam = (key, value) => {
    setParams(prev => {
      const newParams = { ...prev, [key]: value };
      
      // For RSMA power ratios, ensure they sum to 1
      if (key === 'commonPowerRatio' || key === 'privatePower1Ratio' || key === 'privatePower2Ratio') {
        const remaining = 1 - value;
        const otherKeys = ['commonPowerRatio', 'privatePower1Ratio', 'privatePower2Ratio'].filter(k => k !== key);
        
        // Distribute remaining power proportionally to other ratios
        const otherSum = otherKeys.reduce((sum, k) => sum + prev[k], 0);
        if (otherSum > 0) {
          otherKeys.forEach(k => {
            newParams[k] = (prev[k] / otherSum) * remaining;
          });
        } else {
          // If others are 0, distribute equally
          otherKeys.forEach(k => {
            newParams[k] = remaining / 2;
          });
        }
      }
      
      // For NOMA, ensure a1 + a2 = 1 (a1 is implicit, a2 is nomaPowerRatio)
      // a1 = 1 - a2 is calculated in runNOMASimulation
      
      return newParams;
    });
  };
  
  // Generate comparison data for SNR sweep
  const generateComparisonData = () => {
    const snrRange = [0, 5, 10, 15, 20, 25, 30]; // dB
    const data = snrRange.map(snr => {
      let testParams = { ...params, SNR: snr };
      
      // If auto-optimize is enabled, use optimized power ratios for each SNR point
      if (params.autoOptimize) {
        const { bestRatios } = optimizeRSMAPower(testParams);
        testParams = { ...testParams, ...bestRatios };
      }
      
      const rsmaResult = runRSMASimulation(testParams);
      const nomaResult = runNOMASimulation(testParams);
      const omaResult = runOMASimulation(testParams);
      return {
        snr: snr,
        rsmaSumRate: rsmaResult.sumRate,
        nomaSumRate: nomaResult.sumRate,
        omaSumRate: omaResult.sumRate,
      };
    });
    
    return data;
  };
  
  const comparisonData = generateComparisonData();
  
  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #f8fafc, #eff6ff)', padding: '32px', fontFamily: 'Arial, sans-serif' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
          <h1 style={{ fontSize: '36px', fontWeight: 'bold', color: '#1f2937', marginBottom: '8px' }}>RSMA vs NOMA vs OMA</h1>
          <p style={{ color: '#4b5563' }}>MISO Downlink System with 2 Users (Perfect/Imperfect CSI, Beamforming)</p>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px' }}>
          <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', padding: '24px' }}>
            <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937', marginBottom: '24px' }}>System Parameters</h2>
            
            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '16px', marginTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#4b5563', marginBottom: '16px' }}>Antenna Configuration</h3>
              
              <Slider 
                label="Base Station Antennas (Nt)" 
                value={params.Nt} 
                onChange={(v) => updateParam('Nt', Math.round(v))}
                min={1} 
                max={4} 
                step={1}
              />
            </div>
            
            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '16px', marginTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#4b5563', marginBottom: '16px' }}>Channel Gains</h3>
              
              <Slider 
                label="User 1 Channel (Near)" 
                value={params.h1} 
                onChange={(v) => updateParam('h1', v)}
                min={0.1} 
                max={1.0} 
                step={0.05}
              />
              
              <Slider 
                label="User 2 Channel (Far)" 
                value={params.h2} 
                onChange={(v) => updateParam('h2', v)}
                min={0.01} 
                max={1.0} 
                step={0.01}
              />
              
              <Slider 
                label="Spatial Angle (degrees)" 
                value={params.spatialAngle} 
                onChange={(v) => updateParam('spatialAngle', v)}
                min={0} 
                max={90} 
                step={5}
                unit="°"
              />
            </div>
            
            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '16px', marginTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#4b5563', marginBottom: '16px' }}>Power & Noise</h3>
              
              <Slider 
                label="Noise Power (σ²)" 
                value={params.noisePower} 
                onChange={(v) => updateParam('noisePower', v)}
                min={0.1} 
                max={10.0} 
                step={0.1}
              />
              
              <Slider 
                label="SNR (dB)" 
                value={params.SNR} 
                onChange={(v) => updateParam('SNR', v)}
                min={0} 
                max={30} 
                step={1}
                unit=" dB"
              />
              
              <Slider 
                label="CSI Estimation Error" 
                value={params.csiError} 
                onChange={(v) => updateParam('csiError', v)}
                min={0.0} 
                max={0.5} 
                step={0.05}
              />
              
              <Slider 
                label="Graph Y-Axis Max (0 = auto)" 
                value={params.graphYMax} 
                onChange={(v) => updateParam('graphYMax', v)}
                min={0.0} 
                max={50.0} 
                step={5.0}
              />
              
              <Slider 
                label="Min Rate Constraint (QoS)" 
                value={params.minRate} 
                onChange={(v) => updateParam('minRate', v)}
                min={0.0} 
                max={2.0} 
                step={0.1}
		unit=" bps/Hz"
              />
            </div>
            
            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '16px', marginTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#4b5563', marginBottom: '16px' }}>RSMA Power Allocation</h3>
              
              <div style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  checked={params.autoOptimize}
                  onChange={(e) => updateParam('autoOptimize', e.target.checked)}
                  style={{ width: '18px', height: '18px', cursor: 'pointer' }}
                />
                <label style={{ fontSize: '14px', fontWeight: '500', color: '#374151', cursor: 'pointer' }}>
                  Auto-optimize Power Ratios
                </label>
              </div>
              
              {!params.autoOptimize && (
                <>
                  <Slider 
                    label="Common Power Ratio (Pc/P)" 
                    value={params.commonPowerRatio} 
                    onChange={(v) => updateParam('commonPowerRatio', v)}
                    min={0.0} 
                    max={1.0} 
                    step={0.05}
                  />
                  
                  <Slider 
                    label="Private Power 1 Ratio (P1/P)" 
                    value={params.privatePower1Ratio} 
                    onChange={(v) => updateParam('privatePower1Ratio', v)}
                    min={0.0} 
                    max={1.0} 
                    step={0.05}
                  />
                  
                  <Slider 
                    label="Private Power 2 Ratio (P2/P)" 
                    value={params.privatePower2Ratio} 
                    onChange={(v) => updateParam('privatePower2Ratio', v)}
                    min={0.0} 
                    max={1.0} 
                    step={0.05}
                  />
                </>
              )}
              
              {params.autoOptimize && optimizedRatios && (
                <div style={{ padding: '12px', backgroundColor: '#ecfdf5', borderRadius: '8px', border: '1px solid #10b981' }}>
                  <div style={{ fontSize: '13px', fontWeight: '600', color: '#065f46', marginBottom: '8px' }}>
                    Optimized Power Ratios:
                  </div>
                  <div style={{ fontSize: '13px', color: '#065f46', marginBottom: '4px' }}>
                    Pc/P: {optimizedRatios.commonPowerRatio.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '13px', color: '#065f46', marginBottom: '4px' }}>
                    P1/P: {optimizedRatios.privatePower1Ratio.toFixed(3)}
                  </div>
                  <div style={{ fontSize: '13px', color: '#065f46' }}>
                    P2/P: {optimizedRatios.privatePower2Ratio.toFixed(3)}
                  </div>
                </div>
              )}
              
              {params.autoOptimize && (
                <div style={{ padding: '12px', backgroundColor: '#eff6ff', borderRadius: '8px', fontSize: '13px', color: '#1e40af' }}>
                  Power ratios are automatically optimized via grid search for maximum sum rate.
                </div>
              )}
            </div>
            
            <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '16px', marginTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#4b5563', marginBottom: '16px' }}>NOMA Power Allocation</h3>
              
              <Slider 
                label="Weak User Power Ratio (a₂)" 
                value={params.nomaPowerRatio} 
                onChange={(v) => updateParam('nomaPowerRatio', v)}
                min={0.5} 
                max={0.9} 
                step={0.05}
              />
            </div>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', padding: '24px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937', marginBottom: '24px' }}>Current Performance (SNR = {params.SNR.toFixed(1)} dB)</h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
                <div style={{ background: 'linear-gradient(to bottom right, #eff6ff, #dbeafe)', borderRadius: '8px', padding: '16px', border: '2px solid #3b82f6' }}>
                  <div style={{ fontSize: '16px', color: '#2563eb', fontWeight: 'bold', marginBottom: '8px' }}>RSMA</div>
                  <div style={{ fontSize: '14px', color: '#1e40af', marginBottom: '4px' }}>User 1 Rate: {results.rsma?.rate1.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '14px', color: '#1e40af', marginBottom: '4px' }}>User 2 Rate: {results.rsma?.rate2.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '14px', color: '#1e40af', marginBottom: '4px' }}>Common Rate: {results.rsma?.commonRate.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '16px', color: '#1e40af', fontWeight: 'bold', marginTop: '8px' }}>Sum Rate: {results.rsma?.sumRate.toFixed(3)} bps/Hz</div>
                </div>
                
                <div style={{ background: 'linear-gradient(to bottom right, #faf5ff, #e9d5ff)', borderRadius: '8px', padding: '16px', border: '2px solid #9333ea' }}>
                  <div style={{ fontSize: '16px', color: '#9333ea', fontWeight: 'bold', marginBottom: '8px' }}>NOMA</div>
                  <div style={{ fontSize: '14px', color: '#7e22ce', marginBottom: '4px' }}>User 1 Rate: {results.noma?.rate1.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '14px', color: '#7e22ce', marginBottom: '4px' }}>User 2 Rate: {results.noma?.rate2.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '16px', color: '#7e22ce', fontWeight: 'bold', marginTop: '8px' }}>Sum Rate: {results.noma?.sumRate.toFixed(3)} bps/Hz</div>
                </div>
                
                <div style={{ background: 'linear-gradient(to bottom right, #f0fdf4, #dcfce7)', borderRadius: '8px', padding: '16px', border: '2px solid #16a34a' }}>
                  <div style={{ fontSize: '16px', color: '#16a34a', fontWeight: 'bold', marginBottom: '8px' }}>OMA</div>
                  <div style={{ fontSize: '14px', color: '#15803d', marginBottom: '4px' }}>User 1 Rate: {results.oma?.rate1.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '14px', color: '#15803d', marginBottom: '4px' }}>User 2 Rate: {results.oma?.rate2.toFixed(3)} bps/Hz</div>
                  <div style={{ fontSize: '16px', color: '#15803d', fontWeight: 'bold', marginTop: '8px' }}>Sum Rate: {results.oma?.sumRate.toFixed(3)} bps/Hz</div>
                </div>
              </div>
              
              <div style={{ padding: '16px', backgroundColor: '#f3f4f6', borderRadius: '8px' }}>
                <div style={{ fontSize: '14px', color: '#374151', marginBottom: '8px' }}>
                  <strong>RSMA Gain over NOMA:</strong> {((results.rsma?.sumRate / results.noma?.sumRate - 1) * 100).toFixed(1)}%
                </div>
                <div style={{ fontSize: '14px', color: '#374151' }}>
                  <strong>RSMA Gain over OMA:</strong> {((results.rsma?.sumRate / results.oma?.sumRate - 1) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            
            <div style={{ backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', padding: '24px' }}>
              <h2 style={{ fontSize: '20px', fontWeight: '600', color: '#1f2937', marginBottom: '24px' }}>Sum Rate vs SNR (dB)</h2>
              
              <div style={{ marginBottom: '16px' }}>
                <canvas ref={(canvas) => {
                  if (canvas) {
                    const ctx = canvas.getContext('2d');
                    const width = canvas.width = canvas.offsetWidth * 2;
                    const height = canvas.height = 300 * 2;
                    ctx.scale(2, 2);
                    
                    const padding = 60;
                    const chartWidth = (width / 2) - padding * 2;
                    const chartHeight = (height / 2) - padding * 2;
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    
                    // Find max value for scaling (use user-specified or auto-scale)
                    const allValues = [
                      ...comparisonData.map(d => d.rsmaSumRate),
                      ...comparisonData.map(d => d.nomaSumRate),
                      ...comparisonData.map(d => d.omaSumRate)
                    ];
                    const maxValue = params.graphYMax > 0 ? params.graphYMax : Math.max(...allValues) * 1.1;
                    
                    // Draw grid lines
                    ctx.strokeStyle = '#e5e7eb';
                    ctx.lineWidth = 1;
                    const numGridLines = 10; // Finer segmentation
                    for (let i = 0; i <= numGridLines; i++) {
                      const y = padding + (chartHeight / numGridLines) * i;
                      ctx.beginPath();
                      ctx.moveTo(padding, y);
                      ctx.lineTo(padding + chartWidth, y);
                      ctx.stroke();
                      
                      // Y-axis labels (show every other label to avoid clutter)
                      if (i % 2 === 0) {
                        ctx.fillStyle = '#6b7280';
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'right';
                        const value = maxValue - (maxValue / numGridLines) * i;
                        ctx.fillText(value.toFixed(2), padding - 10, y + 4);
                      }
                    }
                    
                    // Draw RSMA line
                    ctx.strokeStyle = '#3b82f6';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    comparisonData.forEach((point, i) => {
                      const x = padding + (chartWidth / (comparisonData.length - 1)) * i;
                      const y = padding + chartHeight - (point.rsmaSumRate / maxValue) * chartHeight;
                      if (i === 0) ctx.moveTo(x, y);
                      else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    
                    // Draw NOMA line
                    ctx.strokeStyle = '#9333ea';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    comparisonData.forEach((point, i) => {
                      const x = padding + (chartWidth / (comparisonData.length - 1)) * i;
                      const y = padding + chartHeight - (point.nomaSumRate / maxValue) * chartHeight;
                      if (i === 0) ctx.moveTo(x, y);
                      else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    
                    // Draw OMA line
                    ctx.strokeStyle = '#16a34a';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    comparisonData.forEach((point, i) => {
                      const x = padding + (chartWidth / (comparisonData.length - 1)) * i;
                      const y = padding + chartHeight - (point.omaSumRate / maxValue) * chartHeight;
                      if (i === 0) ctx.moveTo(x, y);
                      else ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                    
                    // Draw X-axis labels
                    ctx.fillStyle = '#6b7280';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    comparisonData.forEach((point, i) => {
                      const x = padding + (chartWidth / (comparisonData.length - 1)) * i;
                      ctx.fillText(point.snr.toString(), x, padding + chartHeight + 20);
                    });
                    
                    // X-axis label
                    ctx.font = '14px Arial';
                    ctx.fontWeight = 'bold';
                    ctx.fillText('SNR (dB)', padding + chartWidth / 2, padding + chartHeight + 40);
                    
                    // Y-axis label
                    ctx.save();
                    ctx.translate(20, padding + chartHeight / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.textAlign = 'center';
                    ctx.fillText('Sum Rate (bps/Hz)', 0, 0);
                    ctx.restore();
                    
                    // Legend
                    ctx.font = '14px Arial';
                    ctx.textAlign = 'left';
                    
                    ctx.fillStyle = '#3b82f6';
                    ctx.fillRect(padding + chartWidth - 200, padding - 30, 20, 3);
                    ctx.fillStyle = '#1f2937';
                    ctx.fillText('RSMA', padding + chartWidth - 175, padding - 25);
                    
                    ctx.fillStyle = '#9333ea';
                    ctx.fillRect(padding + chartWidth - 100, padding - 30, 20, 3);
                    ctx.fillStyle = '#1f2937';
                    ctx.fillText('NOMA', padding + chartWidth - 75, padding - 25);
                    
                    ctx.fillStyle = '#16a34a';
                    ctx.fillRect(padding + chartWidth - 200, padding - 10, 20, 3);
                    ctx.fillStyle = '#1f2937';
                    ctx.fillText('OMA', padding + chartWidth - 175, padding - 5);
                  }
                }} style={{ width: '100%', height: '300px' }} />
              </div>
              
              <div style={{ marginTop: '16px', padding: '16px', backgroundColor: '#f3f4f6', borderRadius: '8px' }}>
                <div style={{ fontSize: '14px', color: '#374151', marginBottom: '8px' }}>
                  <strong>Channel Gap:</strong> {(params.h1 / params.h2).toFixed(1)}x (Near user has {params.h1.toFixed(2)} gain, Far user has {params.h2.toFixed(2)} gain)
                </div>
                <div style={{ fontSize: '14px', color: '#374151', marginBottom: '8px' }}>
                  <strong>Setup:</strong> MISO downlink with {params.csiError > 0 ? 'imperfect' : 'perfect'} CSI. User 1 (Near), User 2 (Far).
                </div>
                <div style={{ fontSize: '14px', color: '#374151' }}>
                  <strong>RSMA Flexibility:</strong> RSMA can replicate OMA (Pc=0) and NOMA (P2=0) by adjusting power allocation, providing a unified framework.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
