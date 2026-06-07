//! # markov-chain-rs
//!
//! Markov chain algorithms in pure Rust:
//! - **Discrete-time Markov chains** with transition matrices
//! - **Transition matrix** operations
//! - **Stationary distribution** computation
//! - **Absorbing chain** analysis
//! - **Hidden Markov Models** (HMM) with Viterbi and Forward algorithms

/// Discrete-time Markov chain.
pub mod chain;
/// Transition matrix representation and operations.
pub mod transition;
/// Stationary distribution computation.
pub mod stationary;
/// Absorbing Markov chain analysis.
pub mod absorbing;
/// Hidden Markov Model (Hiterbi, Forward algorithm).
pub mod hmm;
