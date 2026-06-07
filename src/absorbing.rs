//! Absorbing Markov chain analysis.

use crate::transition::TransitionMatrix;

/// Identify absorbing states (states that only transition to themselves).
pub fn find_absorbing_states(tm: &TransitionMatrix) -> Vec<usize> {
    (0..tm.n())
        .filter(|&i| (tm.get(i, i) - 1.0).abs() < 1e-10)
        .collect()
}

/// Check if a state is absorbing.
pub fn is_absorbing(tm: &TransitionMatrix, state: usize) -> bool {
    (tm.get(state, state) - 1.0).abs() < 1e-10
}

/// Compute the fundamental matrix N = (I - Q)^(-1) for absorbing chains.
pub fn fundamental_matrix(tm: &TransitionMatrix) -> Option<Vec<Vec<f64>>> {
    let absorbing = find_absorbing_states(tm);
    let transient: Vec<usize> = (0..tm.n()).filter(|i| !absorbing.contains(i)).collect();
    let t = transient.len();
    if t == 0 {
        return None;
    }

    let mut aug = vec![vec![0.0; 2 * t]; t];
    for (i, &si) in transient.iter().enumerate() {
        for (j, &sj) in transient.iter().enumerate() {
            aug[i][j] = if i == j { 1.0 } else { 0.0 } - tm.get(si, sj);
        }
        aug[i][t + i] = 1.0;
    }

    for col in 0..t {
        let mut pivot_row = col;
        for row in (col + 1)..t {
            if aug[row][col].abs() > aug[pivot_row][col].abs() {
                pivot_row = row;
            }
        }
        if aug[pivot_row][col].abs() < 1e-12 {
            return None;
        }
        aug.swap(col, pivot_row);
        let pivot = aug[col][col];
        aug[col].iter_mut().for_each(|v| *v /= pivot);
        for row in 0..t {
            if row != col {
                let factor = aug[row][col];
                let col_row = aug[col].clone();
                for (val, c) in aug[row].iter_mut().zip(col_row) {
                    *val -= factor * c;
                }
            }
        }
    }

    let mut n_matrix = vec![vec![0.0; t]; t];
    for (i, n_row) in n_matrix.iter_mut().enumerate() {
        for (j, val) in n_row.iter_mut().enumerate() {
            *val = aug[i][t + j];
        }
    }
    Some(n_matrix)
}

/// Compute expected number of steps to absorption from each transient state.
pub fn expected_steps_to_absorption(tm: &TransitionMatrix) -> Option<Vec<f64>> {
    let n_matrix = fundamental_matrix(tm)?;
    Some(n_matrix.iter().map(|row| row.iter().sum()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_absorbing() {
        let tm = TransitionMatrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.3, 0.3, 0.4],
        ]).unwrap();
        assert_eq!(find_absorbing_states(&tm), vec![0]);
    }

    #[test]
    fn test_is_absorbing() {
        let tm = TransitionMatrix::new(vec![
            vec![1.0, 0.0],
            vec![0.5, 0.5],
        ]).unwrap();
        assert!(is_absorbing(&tm, 0));
        assert!(!is_absorbing(&tm, 1));
    }

    #[test]
    fn test_fundamental_matrix() {
        let tm = TransitionMatrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5],
            vec![0.5, 0.5, 0.0],
        ]).unwrap();
        let n = fundamental_matrix(&tm);
        assert!(n.is_some());
        assert_eq!(n.unwrap().len(), 2);
    }

    #[test]
    fn test_expected_steps() {
        let tm = TransitionMatrix::new(vec![
            vec![1.0, 0.0],
            vec![0.5, 0.5],
        ]).unwrap();
        let expected = expected_steps_to_absorption(&tm);
        assert!(expected.is_some());
        assert!((expected.unwrap()[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_all_absorbing() {
        let tm = TransitionMatrix::new(vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ]).unwrap();
        assert_eq!(find_absorbing_states(&tm).len(), 2);
    }

    #[test]
    fn test_no_absorbing() {
        let tm = TransitionMatrix::new(vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ]).unwrap();
        assert!(find_absorbing_states(&tm).is_empty());
    }
}
