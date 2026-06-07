//! Hidden Markov Model (HMM) with Viterbi and Forward algorithms.

/// A Hidden Markov Model.
pub struct HMM {
    /// Number of hidden states.
    pub n: usize,
    /// Number of observation symbols.
    pub m: usize,
    /// `transition[i][j]` = P(state j | state i)
    pub transition: Vec<Vec<f64>>,
    /// `emission[i][k]` = P(observation k | state i)
    pub emission: Vec<Vec<f64>>,
    /// `initial[i]` = P(start in state i)
    pub initial: Vec<f64>,
}

impl HMM {
    /// Create a new HMM.
    pub fn new(
        transition: Vec<Vec<f64>>,
        emission: Vec<Vec<f64>>,
        initial: Vec<f64>,
    ) -> Result<Self, &'static str> {
        let n = transition.len();
        if initial.len() != n {
            return Err("Initial distribution length mismatch");
        }
        if emission.len() != n {
            return Err("Emission matrix rows must match state count");
        }
        Ok(Self {
            n,
            m: emission[0].len(),
            transition,
            emission,
            initial,
        })
    }

    /// Viterbi algorithm: find the most likely sequence of hidden states.
    pub fn viterbi(&self, observations: &[usize]) -> (Vec<usize>, f64) {
        let t_len = observations.len();
        if t_len == 0 {
            return (vec![], 0.0);
        }

        let neg_inf = f64::NEG_INFINITY;
        let mut viterbi = vec![vec![neg_inf; self.n]; t_len];
        let mut backptr = vec![vec![0usize; self.n]; t_len];

        // Initialize
        for (s, vit_row) in viterbi[0].iter_mut().enumerate() {
            *vit_row = self.initial[s].ln() + self.emission[s][observations[0]].ln();
        }

        // Recursion
        for (t_idx, obs) in observations.iter().enumerate().skip(1) {
            for s in 0..self.n {
                let emit = self.emission[s][*obs].ln();
                let mut best = neg_inf;
                let mut best_prev = 0;
                for (ps, prev_vit) in viterbi[t_idx - 1].iter().enumerate() {
                    let score = prev_vit + self.transition[ps][s].ln();
                    if score > best {
                        best = score;
                        best_prev = ps;
                    }
                }
                viterbi[t_idx][s] = best + emit;
                backptr[t_idx][s] = best_prev;
            }
        }

        // Termination
        let mut best_score = neg_inf;
        let mut best_state = 0;
        for (s, &score) in viterbi[t_len - 1].iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_state = s;
            }
        }

        // Backtrace
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_state;
        for t_idx in (1..t_len).rev() {
            path[t_idx - 1] = backptr[t_idx][path[t_idx]];
        }

        (path, best_score)
    }

    /// Forward algorithm: compute P(observations | model).
    pub fn forward_log_likelihood(&self, observations: &[usize]) -> f64 {
        let t_len = observations.len();
        if t_len == 0 {
            return 0.0;
        }

        let neg_inf = f64::NEG_INFINITY;
        let mut alpha: Vec<f64> = self.initial.iter().enumerate()
            .map(|(s, init)| init.ln() + self.emission[s][observations[0]].ln())
            .collect();

        for obs in observations.iter().skip(1) {
            let mut new_alpha = vec![neg_inf; self.n];
            for (s, na) in new_alpha.iter_mut().enumerate() {
                let emit = self.emission[s][*obs].ln();
                let mut sum = neg_inf;
                for (ps, &a) in alpha.iter().enumerate() {
                    let score = a + self.transition[ps][s].ln();
                    sum = log_add(sum, score);
                }
                *na = sum + emit;
            }
            alpha = new_alpha;
        }

        let mut total = neg_inf;
        for a in &alpha {
            total = log_add(total, *a);
        }
        total
    }
}

/// Log-space addition: log(a + b) given log(a) and log(b).
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (larger, smaller) = if a > b { (a, b) } else { (b, a) };
    larger + (1.0 + (smaller - larger).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_hmm() -> HMM {
        HMM::new(
            vec![vec![0.7, 0.3], vec![0.4, 0.6]],
            vec![vec![0.9, 0.1], vec![0.2, 0.8]],
            vec![0.6, 0.4],
        ).unwrap()
    }

    #[test]
    fn test_hmm_creation() {
        let hmm = simple_hmm();
        assert_eq!(hmm.n, 2);
        assert_eq!(hmm.m, 2);
    }

    #[test]
    fn test_viterbi() {
        let hmm = simple_hmm();
        let obs = vec![0, 0, 1];
        let (path, score) = hmm.viterbi(&obs);
        assert_eq!(path.len(), 3);
        assert!(score.is_finite());
        assert!(path.iter().all(|&s| s < 2));
    }

    #[test]
    fn test_viterbi_all_same() {
        let hmm = simple_hmm();
        let (path, _) = hmm.viterbi(&[0, 0, 0, 0]);
        assert_eq!(path, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_viterbi_obs1() {
        let hmm = simple_hmm();
        let (path, _) = hmm.viterbi(&[1, 1, 1]);
        assert!(path.iter().all(|&s| s == 1));
    }

    #[test]
    fn test_forward_likelihood() {
        let hmm = simple_hmm();
        let ll = hmm.forward_log_likelihood(&[0, 0, 1]);
        assert!(ll.is_finite());
        assert!(ll < 0.0);
    }

    #[test]
    fn test_forward_longer_less_likely() {
        let hmm = simple_hmm();
        let ll1 = hmm.forward_log_likelihood(&[0]);
        let ll2 = hmm.forward_log_likelihood(&[0, 0]);
        assert!(ll2 < ll1);
    }

    #[test]
    fn test_viterbi_empty() {
        let hmm = simple_hmm();
        let (path, _) = hmm.viterbi(&[]);
        assert!(path.is_empty());
    }

    #[test]
    fn test_forward_empty() {
        let hmm = simple_hmm();
        let ll = hmm.forward_log_likelihood(&[]);
        assert!((ll - 0.0).abs() < 1e-10);
    }
}
