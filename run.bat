@echo off
REM =============================================================================
REM Cross-Group Divergence Attack — Experiment Launcher (Windows)
REM Reproduces all 5 experiments from the report.
REM =============================================================================
cd /d "%~dp0codes"

echo =============================================
echo  Cross-Group Divergence Attack Experiments
echo  Built on Fan et al. (NeurIPS 2021^)
echo  Reproducing ^& extending Fang et al. (WWW 2025^)
echo =============================================
echo.

REM ------------------------------------------------------------------
REM Experiment 1: Baseline (CartPole, no attack, no ensemble)
REM ------------------------------------------------------------------
echo [1/5] Baseline: CartPole, No Attack, No Ensemble
echo       Expected: test reward ~500
python run.py --env_name CartPole-v1 --num_worker 30 --num_Byzantine 0 --no_tb --no_saving
echo.

REM ------------------------------------------------------------------
REM Experiment 2: Normalized Attack vs single-group FedPG-BR
REM ------------------------------------------------------------------
echo [2/5] Normalized Attack vs FedPG-BR (CartPole^)
echo       Expected: test reward ~432 (attack partially effective^)
python run.py --env_name CartPole-v1 --num_worker 30 --num_Byzantine 9 --attack_type normalized-attack --FedPG_BR --no_tb --no_saving
echo.

REM ------------------------------------------------------------------
REM Experiment 3: Normalized Attack vs Ensemble Defense
REM ------------------------------------------------------------------
echo [3/5] Normalized Attack vs Ensemble Defense (CartPole^)
echo       Expected: test reward ~500 (defense holds^)
python run.py --env_name CartPole-v1 --num_worker 30 --num_Byzantine 9 --attack_type normalized-attack --FedPG_BR --ensemble --num_groups 5 --no_tb --no_saving
echo.

REM ------------------------------------------------------------------
REM Experiment 4: Cross-Group Divergence Attack vs Ensemble
REM ------------------------------------------------------------------
echo [4/5] Cross-Group Divergence Attack vs Ensemble (LunarLander^)
echo       Expected: test reward ~152 (defense broken^)
python run.py --env_name LunarLander-v2 --num_worker 30 --num_Byzantine 10 --attack_type divergence-attack --target_action_mode group-mod --FedPG_BR --ensemble --num_groups 5 --no_tb --no_saving
echo.

REM ------------------------------------------------------------------
REM Experiment 5: SecAgg Cryptographic Experiment
REM ------------------------------------------------------------------
echo [5/5] SecAgg + Divergence Attack (CartPole^)
echo       Expected: N_good=30 (filtering disabled by encryption^)
python run.py --env_name CartPole-v1 --num_worker 30 --num_Byzantine 9 --attack_type divergence-attack --FedPG_BR --use_secagg --no_tb --no_saving
echo.

echo =============================================
echo  All experiments complete.
echo =============================================
pause
