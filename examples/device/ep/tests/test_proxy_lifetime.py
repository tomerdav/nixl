# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU proxy retirement test; run directly with the candidate EP package."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import nixl_ep

    torch.cuda.set_device(0)
    store = dist.TCPStore(
        "127.0.0.1", port, is_master=False, timeout=timedelta(seconds=60)
    )

    def barrier(name):
        store.set(f"{name}/{rank}", "ready")
        store.wait([f"{name}/0", f"{name}/1"])

    tokens, hidden, experts = 8, 2560, 4
    size = nixl_ep.Buffer.get_rdma_size_hint(tokens, hidden, 2, experts)
    buffer = nixl_ep.Buffer(
        rank=rank,
        disable_ll_nvlink=True,
        explicitly_destroy=True,
        tcp_store_group=store,
        timeout_ms=5000,
    )
    buffer.update_memory_buffers(2, 2, size)
    x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device="cuda")

    def dispatch(connected):
        selections = [0, 2] if connected else [2 * rank]
        indices = torch.tensor(selections, dtype=nixl_ep.topk_idx_t, device="cuda")
        indices = indices.repeat(tokens, 1)
        weights = torch.full(
            indices.shape, 1 / len(selections), dtype=torch.float32, device="cuda"
        )
        received, counts, handle, event, _ = buffer.dispatch(
            x, indices, tokens, experts, use_fp8=False, async_finish=True
        )
        return received, counts, handle, event, indices, weights

    def check(received, counts, connected):
        assert counts[0].item() == tokens * (2 if connected else 1)
        assert counts[1].item() == 0
        values = received[0, : counts[0].item()]
        for source in [0, 1] if connected else [rank]:
            assert (values[:, 0] == source + 1).sum().item() == tokens
        assert torch.equal(values.amin(dim=1), values.amax(dim=1))

    for round_id in range(3):
        barrier(f"connect-{round_id}")
        buffer.connect_ranks([1 - rank])
        received, counts, handle, event, indices, weights = dispatch(True)
        event.current_stream_wait()
        check(received, counts, True)
        combined, event, _ = buffer.combine(
            received.clone(), indices, weights, handle, async_finish=True
        )
        event.current_stream_wait()
        assert torch.equal(combined, x)
        barrier(f"disconnect-{round_id}")
        received, counts, _, _, _, _ = dispatch(True)
        # No explicit event wait: disconnect must synchronize and retire this work.
        buffer.disconnect_ranks([1 - rank])
        check(received, counts, True)
        received, counts, _, event, _, _ = dispatch(False)
        event.current_stream_wait()
        check(received, counts, False)
        print(
            f"rank={rank} round={round_id}: remove, surviving progress, re-add PASS",
            flush=True,
        )
    barrier("done")
    buffer.destroy()


if __name__ == "__main__":
    os.environ["NIXL_EP_DEVICE_MODE"] = "proxy"
    port = int(os.environ.get("NIXL_TEST_PORT", "29673"))
    server = dist.TCPStore(
        "127.0.0.1",
        port,
        is_master=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=60),
    )
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
