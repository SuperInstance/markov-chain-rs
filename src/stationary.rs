//! Stationary distribution computation.

use crate::transition::TransitionMatrix;

/// Compute the stationary distribution by power iteration.
///
/// Iterates `v = v * P` until convergence (max iterations or tolerance).
/// Returns the stationary distribution π where π = πP.
pub fn stationary_distribution(tm: &TransitionMatrix, max_iter: u32, tolerance: f64) -> Vec<f64> {
    let n = tm.n();
    let mut v = vec![1.0 / n as f64; n];

    for _ in 0..max_iter {
        let new_v = tm.apply_to_vector(&v);
        let diff = v.iter()
            .zip(new_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        v = new_v;
        if diff < tolerance {
            break;
        }
    }
    v
}

/// Verify that a vector is a valid stationary distribution for the given transition matrix.
///
/// Checks: π = πP, all πᵢ ≥ 0, Σπᵢ = 1.
pub fn is_stationary(tm: &TransitionMatrix, pi: &[f64], tolerance: f64) -> bool {
    let n = tm.n();
    if pi.len() != n {
        return false;
    }
    if pi.iter().any(|&x| x < -tolerance) {
        return false;
    }
    let sum: f64 = pi.iter().sum();
    if (sum - 1.0).abs() > tolerance {
        return false;
    }
    let pi_p = tm.apply_to_vector(pi);
    pi.iter().zip(pi_p.iter()).all(|(a, b)| (a - b).abs() <= tolerance)
}

/// Compute stationary distribution with high precision.
pub fn stationary_direct(tm: &TransitionMatrix) -> Vec<f64> {
    stationary_distribution(tm, 10000, 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stationary_symmetric() {
        let tm = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap();
        let pi = stationary_distribution(&tm, 1000, 1e-10);
        assert!((pi[0] - 0.5).abs() < 1e-6);
        assert!((pi[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_stationary_asymmetric() {
        let tm = TransitionMatrix::new(vec![vec![0.7, 0.3], vec![0.4, 0.6]]).unwrap();
        let pi = stationary_distribution(&tm, 1000, 1e-10);
        assert!((pi[0] - 4.0 / 7.0).abs() < 1e-4);
        assert!((pi[1] - 3.0 / 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_is_stationary() {
        let tm = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap();
        assert!(is_stationary(&tm, &[0.5, 0.5], 1e-9));
    }

    #[test]
    fn test_is_not_stationary() {
        let tm = TransitionMatrix::new(vec![vec![0.7, 0.3], vec![0.4, 0.6]]).unwrap();
        assert!(!is_stationary(&tm, &[0.5, 0.5], 1e-9));
    }

    #[test]
    fn test_stationary_three_states() {
        let tm = TransitionMatrix::new(vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ]).unwrap();
        let pi = stationary_distribution(&tm, 10000, 1e-10);
        assert!((pi[0] - 1.0 / 3.0).abs() < 1e-4);
        assert!((pi[1] - 1.0 / 3.0).abs() < 1e-4);
        assert!((pi[2] - 1.0 / 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_stationary_identity() {
        let tm = TransitionMatrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let pi = stationary_distribution(&tm, 1000, 1e-10);
        assert!(is_stationary(&tm, &pi, 1e-6));
    }
}
