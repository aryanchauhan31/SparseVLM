import torch
class SparsityScheduler:

    def __init__(self, n_vis_max: int, n_buckets: int = 10, min_tokens: int = 32):
        self.n_vis_max = n_vis_max
        self.n_buckets = n_buckets
        self.min_tokens = min_tokens
        self.buckets = self._compute_buckets()
        self._graphs = {}
        self._static_inputs = {}
        self._static_outputs = {}
        self._warmed_up = False

    def _compute_buckets(self) -> list:
        step = (self.n_vis_max - self.min_tokens) / self.n_buckets
        buckets = [int(self.min_tokens + i * step) for i in range(self.n_buckets)]
        buckets[-1] = self.n_vis_max
        return sorted(set(buckets))

    def snap_to_bucket(self, n_vis: int) -> int:
        for b in self.buckets:
            if b >= n_vis:
                return b
        return self.n_vis_max

    def get_bucket_idx(self, n_vis: int) -> int:
        return self.buckets.index(self.snap_to_bucket(n_vis))

    def warmup(self, model_forward_fn, sample_inputs_fn, n_warmup: int = 3):
        if not torch.cuda.is_available():
            print("[SparsityScheduler] CUDA not available — skipping.")
            return

        for idx, n_vis in enumerate(self.buckets):
            static_inputs = sample_inputs_fn(n_vis)
            for _ in range(n_warmup):
                model_forward_fn(static_inputs)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = model_forward_fn(static_inputs)

            self._graphs[idx]        = g
            self._static_inputs[idx] = static_inputs
            self._static_outputs[idx] = static_output

        self._warmed_up = True
        print(f"[SparsityScheduler] Captured graphs for {len(self.buckets)} buckets.")

    def replay(self, bucket_idx: int, new_inputs: dict) -> torch.Tensor:
        if not self._warmed_up:
            raise RuntimeError("Call warmup() first.")
        for key, tensor in new_inputs.items():
            if key in self._static_inputs[bucket_idx]:
                self._static_inputs[bucket_idx][key].copy_(tensor)
        self._graphs[bucket_idx].replay()
        return self._static_outputs[bucket_idx]

    def summary(self) -> str:
        return (
            f"SparsityScheduler: {len(self.buckets)} buckets\n"
            f"  Token counts: {self.buckets}\n"
            f"  Warmed up: {self._warmed_up}"
        )


def make_scheduler(n_vis_max: int, n_buckets: int = 10, min_tokens: int = 32):
    return SparsityScheduler(n_vis_max, n_buckets, min_tokens)
