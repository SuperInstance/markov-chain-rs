//! Discrete-time Markov chain.

use crate::transition::TransitionMatrix;

/// A discrete-time Markov chain.
///
/// Defined by its transition matrix and an initial state distribution.
pub struct MarkovChain {
    /// Transition matrix P.
    pub transition: TransitionMatrix,
    /// Initial state distribution.
    pub initial: Vec<f64>,
}

impl MarkovChain {
    /// Create a new Markov chain with transition matrix and initial distribution.
    pub fn new(transition: TransitionMatrix, initial: Vec<f64>) -> Result<Self, &'static str> {
        if initial.len() != transition.n() {
            return Err("Initial distribution length must match number of states");
        }
        let sum: f64 = initial.iter().sum();
        if (sum - 1.0).abs() > 1e-9 {
            return Err("Initial distribution must sum to 1.0");
        }
        Ok(Self { transition, initial })
    }

    /// Number of states.
    pub fn n(&self) -> usize {
        self.transition.n()
    }

    /// Compute state distribution after `steps` transitions.
    pub fn distribution_after(&self, steps: u32) -> Vec<f64> {
        let mut dist = self.initial.clone();
        for _ in 0..steps {
            dist = self.transition.apply_to_vector(&dist);
        }
        dist
    }

    /// Get the n-step transition matrix.
    pub fn n_step_matrix(&self, n: u32) -> TransitionMatrix {
        self.transition.pow(n)
    }

    /// Simulate one step from state `s`, returning the next state.
    pub fn step_from(&self, s: usize, rng: f64) -> usize {
        let row = self.transition.row(s);
        let mut cumsum = 0.0;
        for (j, &p) in row.iter().enumerate() {
            cumsum += p;
            if rng < cumsum {
                return j;
            }
        }
        row.len() - 1
    }

    /// Simulate a path of length `steps` starting from initial distribution.
    pub fn simulate(&self, steps: usize, rng_values: &[f64]) -> Vec<usize> {
        assert!(rng_values.len() > steps, "Need at least steps+1 random values");
        let mut path = Vec::with_capacity(steps + 1);

        // Choose initial state
        let mut cumsum = 0.0;
        let mut initial_state = self.initial.len() - 1;
        for (i, &p) in self.initial.iter().enumerate() {
            cumsum += p;
            if rng_values[0] < cumsum {
                initial_state = i;
                break;
            }
        }
        path.push(initial_state);

        for t in 0..steps {
            let next = self.step_from(path[t], rng_values[t + 1]);
            path.push(next);
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_state_chain() -> MarkovChain {
        let tm = TransitionMatrix::new(vec![vec![0.7, 0.3], vec![0.4, 0.6]]).unwrap();
        MarkovChain::new(tm, vec![1.0, 0.0]).unwrap()
    }

    #[test]
    fn test_creation() {
        let mc = two_state_chain();
        assert_eq!(mc.n(), 2);
    }

    #[test]
    fn test_distribution_after_0() {
        let mc = two_state_chain();
        let d = mc.distribution_after(0);
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1]).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_after_1() {
        let mc = two_state_chain();
        let d = mc.distribution_after(1);
        assert!((d[0] - 0.7).abs() < 1e-10);
        assert!((d[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_distribution_after_many() {
        let mc = two_state_chain();
        let d = mc.distribution_after(100);
        assert!((d[0] - 4.0 / 7.0).abs() < 0.01);
        assert!((d[1] - 3.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_step_from() {
        let mc = two_state_chain();
        assert_eq!(mc.step_from(0, 0.0), 0);
        assert_eq!(mc.step_from(0, 0.8), 1);
    }

    #[test]
    fn test_simulate() {
        let mc = two_state_chain();
        let rngs: Vec<f64> = (0..12).map(|i| i as f64 * 0.08).collect();
        let path = mc.simulate(10, &rngs);
        assert_eq!(path.len(), 11);
        assert!(path.iter().all(|&s| s < 2));
    }

    #[test]
    fn test_n_step_matrix() {
        let mc = two_state_chain();
        let m = mc.n_step_matrix(0);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-10);
    }
}
