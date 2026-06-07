//! Transition matrix representation and operations.

/// A row-stochastic transition matrix for a Markov chain.
///
/// Each row sums to 1.0, representing transition probabilities from state i.
#[derive(Clone, Debug)]
pub struct TransitionMatrix {
    /// `data[i][j]` = P(transition from state i to state j)
    data: Vec<Vec<f64>>,
    n: usize,
}

impl TransitionMatrix {
    /// Create from a Vec of rows. Validates row sums ≈ 1.0.
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        let n = data.len();
        for row in &data {
            if row.len() != n {
                return Err("Matrix must be square");
            }
            for &val in row {
                if !(0.0..=1.0).contains(&val) {
                    return Err("Probabilities must be in [0,1]");
                }
            }
            let sum: f64 = row.iter().sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err("Each row must sum to 1.0");
            }
        }
        Ok(Self { data, n })
    }

    /// Create from data without validation (for internal use).
    pub fn new_unchecked(data: Vec<Vec<f64>>) -> Self {
        let n = data.len();
        Self { data, n }
    }

    /// Number of states.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get P(i → j).
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    /// Get row i.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    /// Multiply two transition matrices (composition of transitions).
    pub fn multiply(&self, other: &TransitionMatrix) -> TransitionMatrix {
        let n = self.n;
        let mut result = vec![vec![0.0; n]; n];
        for (i, res_row) in result.iter_mut().enumerate() {
            for (j, val) in res_row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.data[i][k] * other.data[k][j];
                }
                *val = sum;
            }
        }
        TransitionMatrix::new_unchecked(result)
    }

    /// Raise to power `k` by repeated squaring.
    pub fn pow(&self, k: u32) -> TransitionMatrix {
        if k == 0 {
            return self.identity();
        }
        if k == 1 {
            return self.clone();
        }
        let mut result = self.identity();
        let mut base = self.clone();
        let mut exp = k;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.multiply(&base);
            }
            base = base.multiply(&base);
            exp /= 2;
        }
        result
    }

    /// Identity matrix (stays in same state with probability 1).
    pub fn identity(&self) -> TransitionMatrix {
        let mut data = vec![vec![0.0; self.n]; self.n];
        for (i, row) in data.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        TransitionMatrix::new_unchecked(data)
    }

    /// Multiply by a probability vector: returns new distribution after one step.
    pub fn apply_to_vector(&self, v: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; self.n];
        for j in 0..self.n {
            let mut sum = 0.0;
            for (vi, row) in v.iter().zip(&self.data) {
                sum += vi * row[j];
            }
            result[j] = sum;
        }
        result
    }

    /// Get the underlying data.
    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_creation() {
        let m = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.3, 0.7]]);
        assert!(m.is_ok());
        assert_eq!(m.unwrap().n(), 2);
    }

    #[test]
    fn test_invalid_row_sum() {
        let m = TransitionMatrix::new(vec![vec![0.5, 0.3], vec![0.3, 0.7]]);
        assert!(m.is_err());
    }

    #[test]
    fn test_negative_prob() {
        let m = TransitionMatrix::new(vec![vec![-0.1, 1.1], vec![0.3, 0.7]]);
        assert!(m.is_err());
    }

    #[test]
    fn test_identity() {
        let m = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.3, 0.7]]).unwrap();
        let id = m.identity();
        assert!((id.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((id.get(0, 1)).abs() < 1e-10);
    }

    #[test]
    fn test_multiply() {
        let m = TransitionMatrix::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let result = m.multiply(&m);
        assert!((result.get(0, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pow() {
        let m = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap();
        let m3 = m.pow(3);
        assert!((m3.get(0, 0) - 0.5).abs() < 1e-10);
        assert!((m3.get(1, 0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_vector() {
        let m = TransitionMatrix::new(vec![vec![0.5, 0.5], vec![0.5, 0.5]]).unwrap();
        let v = vec![1.0, 0.0];
        let r = m.apply_to_vector(&v);
        assert!((r[0] - 0.5).abs() < 1e-10);
        assert!((r[1] - 0.5).abs() < 1e-10);
    }
}
