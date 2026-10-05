# Dispatch A/B settings, sourced by every script here. Everything else comes from the ep-mars kit.
D=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/tdavidor
# Lustre (cluster storage policy): only the kits, the source tarball and one packed results tarball per job.
# Builds, the extracted tree and per-run logs stay on node-local disk.
export AB_DIR=${AB_DIR:-$D/dispatch-ab}
# Node-local scratch: /raid/scratch per the cluster's storage best practices; the others are fallbacks if it
# is not writable. Per job, removed when the job ends.
export AB_LOCAL_CANDIDATES=${AB_LOCAL_CANDIDATES:-/raid/scratch /tmp}
# The ep-mars kit (env-dfw.sh, env.sh, run_node.sh, mk_plans.py, ...), shipped in the A/B bundle.
export EP_KIT=${EP_KIT:-$AB_DIR/ep-kit}
export AB_TAR=${AB_TAR:-$AB_DIR/nixl-src-ef197c55.tar}
export AB_REPS=${AB_REPS:-7}
export AB_RANKS=${AB_RANKS:-8}
export AB_NVL=${AB_NVL:-nvl nonvl}
export AB_MODES=${AB_MODES:-direct proxy}
export EP_KINETO=${EP_KINETO:-1}
# Direct mode with NVLink: with rc_gda allowed, same-node peers get a GDA lane and nixlGetPtr returns null,
# so EP falls back to RDMA. Excluding rc_gda leaves cuda_ipc for them. Single node only: every peer is local.
export AB_DIRECT_NVL_TLS=${AB_DIRECT_NVL_TLS-^rc_gda}
