#!/bin/bash
# =============================================================================
# Cross-Group Divergence Attack — Experiment Launcher
# Reproduces all 5 experiments from the report.
# =============================================================================
set -e
cd "$(dirname "$0")/codes"

echo "============================================="
echo " Cross-Group Divergence Attack Experiments"
echo " Built on Fan et al. (NeurIPS 2021)"
echo " Reproducing & extending Fang et al. (WWW 2025)"
echo "============================================="
echo ""

# ------------------------------------------------------------------
# Experiment 1: Baseline (CartPole, no attack, no ensemble)
# ------------------------------------------------------------------
echo "[1/4] Baseline: CartPole, No Attack, No Ensemble"
echo "      Expected: test reward ~500"
python run.py \
    --env_name CartPole-v1 \
    --num_worker 30 \
    --num_Byzantine 0 \
    --no_tb --no_saving
echo ""

# ------------------------------------------------------------------
# Experiment 2: Normalized Attack vs single-group FedPG-BR
# ------------------------------------------------------------------
echo "[2/4] Normalized Attack vs FedPG-BR (CartPole)"
echo "      Expected: test reward ~432 (attack partially effective)"
python run.py \
    --env_name CartPole-v1 \
    --num_worker 30 \
    --num_Byzantine 9 \
    --attack_type normalized-attack \
    --FedPG_BR \
    --no_tb --no_saving
echo ""

# ------------------------------------------------------------------
# Experiment 3: Normalized Attack vs Ensemble Defense
# ------------------------------------------------------------------
echo "[3/4] Normalized Attack vs Ensemble Defense (CartPole)"
echo "      Expected: test reward ~500 (defense holds)"
python run.py \
    --env_name CartPole-v1 \
    --num_worker 30 \
    --num_Byzantine 9 \
    --attack_type normalized-attack \
    --FedPG_BR \
    --ensemble --num_groups 5 \
    --no_tb --no_saving
echo ""

# ------------------------------------------------------------------
# Experiment 4: Cross-Group Divergence Attack vs Ensemble
# ------------------------------------------------------------------
echo "[4/4] Cross-Group Divergence Attack vs Ensemble (LunarLander)"
echo "      Expected: test reward ~152 (defense broken)"
python run.py \
    --env_name LunarLander-v3 \
    --num_worker 30 \
    --num_Byzantine 4 \
    --attack_type divergence-attack \
    --FedPG_BR \
    --ensemble --num_groups 5 \
    --no_tb --no_saving
echo ""

echo "============================================="
echo " All experiments complete."
echo "============================================="
