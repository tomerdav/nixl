# Dispatch A/B settings, sourced by every script here. Everything else comes from the ep-mars kit.
D=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor
export AB_DIR=${AB_DIR:-$D/dispatch-ab}
# The ep-mars kit (env-dfw.sh, env.sh, run_node.sh, mk_plans.py, ...), shipped in the A/B bundle.
export EP_KIT=${EP_KIT:-$AB_DIR/ep-kit}
export AB_REPS=${AB_REPS:-7}
export AB_RANKS=${AB_RANKS:-8}
export AB_NVL=${AB_NVL:-nvl nonvl}
export AB_MODES=${AB_MODES:-direct proxy}
export EP_KINETO=${EP_KINETO:-1}
