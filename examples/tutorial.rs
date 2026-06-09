//! Tutorial: markov-chain-rs — Markov chains, HMMs, and stochastic processes

use markov_chain_rs::{
    transition::TransitionMatrix,
    chain::MarkovChain,
    stationary::{stationary_distribution, is_stationary, stationary_direct},
    absorbing::{find_absorbing_states, expected_steps_to_absorption},
    hmm::HMM,
};

fn main() {
    println!("=== Markov Chain Tutorial ===\n");

    // Part 1: Transition matrix
    println!("Part 1: Transition matrix");
    let tm = TransitionMatrix::new(vec![
        vec![0.7, 0.2, 0.1],
        vec![0.3, 0.5, 0.2],
        vec![0.1, 0.3, 0.6],
    ]).unwrap();
    println!("  3-state transition matrix created");
    println!();

    // Part 2: Markov chain evolution
    println!("Part 2: Chain evolution");
    let chain = MarkovChain::new(tm.clone(), vec![1.0, 0.0, 0.0]).unwrap();
    let dist5 = chain.distribution_after(5);
    let dist50 = chain.distribution_after(50);
    println!("  After 5 steps:  {:?}", dist5.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("  After 50 steps: {:?}", dist50.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!();

    // Part 3: Stationary distribution
    println!("Part 3: Stationary distribution");
    let pi = stationary_distribution(&tm, 1000, 1e-10);
    println!("  Iterative: {:?}", pi.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    let pi_direct = stationary_direct(&tm);
    println!("  Direct:    {:?}", pi_direct.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>());
    println!("  Is stationary: {}", is_stationary(&tm, &pi, 1e-6));
    println!();

    // Part 4: Simulation
    println!("Part 4: Simulation");
    let rng_values: Vec<f64> = (0..20).map(|i| ((i as f64 * 0.049) + 0.001).min(0.999)).collect();
    let path = chain.simulate(20, &rng_values);
    println!("  Path: {:?}", path);
    println!();

    // Part 5: Absorbing states
    println!("Part 5: Absorbing Markov chain");
    let absorbing_tm = TransitionMatrix::new(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.3, 0.4, 0.3],
        vec![0.2, 0.3, 0.5],
    ]).unwrap();
    let absorbing = find_absorbing_states(&absorbing_tm);
    println!("  Absorbing states: {:?}", absorbing);
    if let Some(expected) = expected_steps_to_absorption(&absorbing_tm) {
        println!("  Expected steps: {:?}", expected.iter().map(|v| format!("{:.2}", v)).collect::<Vec<_>>());
    }
    println!();

    // Part 6: Hidden Markov Model (Viterbi)
    println!("Part 6: Hidden Markov Model");
    let hmm = HMM::new(
        vec![vec![0.7, 0.3], vec![0.4, 0.6]],        // transition
        vec![vec![0.1, 0.4, 0.5], vec![0.7, 0.2, 0.1]], // emission
        vec![0.5, 0.5],                              // initial
    ).unwrap();
    let observations = vec![0, 0, 1, 2, 0];
    let (path, prob) = hmm.viterbi(&observations);
    println!("  Observations: {:?}", observations);
    println!("  Viterbi path: {:?}", path);
    println!("  Log prob: {:.4}", prob);
    let ll = hmm.forward_log_likelihood(&observations);
    println!("  Forward LL: {:.4}", ll);
}
