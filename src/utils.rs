use rustfft::{num_complex::Complex, FftPlanner};

pub fn std_dev(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}

pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
}

pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Degree-3 polynomial least-squares fit via normal equations.
/// Returns `[c0, c1, c2, c3]`: `y ≈ c0 + c1*x + c2*x² + c3*x³`.
pub fn polyfit_3(x: &[f64], y: &[f64]) -> [f64; 4] {
    let n = x.len();
    assert!(n >= 4, "polyfit_3 requires at least 4 points");

    // A^T A c = A^T y, where A = Vandermonde [1, x, x², x³]
    let mut ata = [[0.0f64; 4]; 4];
    let mut aty = [0.0f64; 4];

    for k in 0..n {
        let xi = x[k];
        let yi = y[k];
        let powers = [1.0, xi, xi * xi, xi * xi * xi];
        for i in 0..4 {
            aty[i] += powers[i] * yi;
            for j in 0..4 {
                ata[i][j] += powers[i] * powers[j];
            }
        }
    }

    // Gaussian elimination with partial pivoting
    let mut aug = [[0.0f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = ata[i][j];
        }
        aug[i][4] = aty[i];
    }

    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for (row, aug_row) in aug.iter().enumerate().skip(col + 1) {
            if aug_row[col].abs() > max_val {
                max_val = aug_row[col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-15 {
            return [0.0; 4];
        }

        let inv_pivot = 1.0 / pivot;
        for val in &mut aug[col][col..5] {
            *val *= inv_pivot;
        }
        for row in 0..4 {
            if row != col {
                let factor = aug[row][col];
                let col_vals = aug[col];
                for (dest, &src) in aug[row][col..].iter_mut().zip(col_vals[col..].iter()) {
                    *dest -= factor * src;
                }
            }
        }
    }

    [aug[0][4], aug[1][4], aug[2][4], aug[3][4]]
}

pub fn polyval_3(coefs: &[f64; 4], x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&xi| coefs[0] + coefs[1] * xi + coefs[2] * xi * xi + coefs[3] * xi * xi * xi)
        .collect()
}

/// Returns `(magnitudes, frequencies)` from real FFT; frequencies in Hz given sample spacing `d` seconds.
pub fn rfft_magnitudes(data: &[f64], d: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft.process(&mut buffer);

    let n_rfft = n / 2 + 1;
    let magnitudes: Vec<f64> = buffer[..n_rfft]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    let freqs: Vec<f64> = (0..n_rfft).map(|k| k as f64 / (n as f64 * d)).collect();

    (magnitudes, freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_dev() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&data);
        assert!((sd - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_median_odd() {
        let data = [3.0, 1.0, 2.0];
        assert!((median(&data) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert!((median(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_polyfit_3_linear() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 + 3.0 * xi).collect();
        let c = polyfit_3(&x, &y);
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!((c[1] - 3.0).abs() < 1e-6);
        assert!(c[2].abs() < 1e-6);
        assert!(c[3].abs() < 1e-6);
    }

    #[test]
    fn test_rfft_magnitudes_dc() {
        let data = vec![1.0; 8];
        let (mags, freqs) = rfft_magnitudes(&data, 1.0);
        assert_eq!(mags.len(), 5);
        assert!((mags[0] - 8.0).abs() < 1e-10);
        assert!((freqs[0]).abs() < 1e-10);
        for &m in &mags[1..] {
            assert!(m.abs() < 1e-10);
        }
    }
}
