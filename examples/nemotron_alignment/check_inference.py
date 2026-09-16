"""Compare realized decode tokens with full-prefill scoring and batch-one replay."""

import json
import os
import random
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main(dp_rank=0, dp_size=1, dp_port=None):
    if dp_size > 1:
        local_rank = dp_rank
        local_size = int(os.environ.get("NEMOTRON_DP_LOCAL", str(dp_size)))
        dp_rank += int(os.environ.get("NEMOTRON_NODE_RANK", "0")) * local_size
        os.environ.update(
            VLLM_DP_RANK=str(dp_rank),
            VLLM_DP_RANK_LOCAL=str(local_rank),
            VLLM_DP_SIZE=str(dp_size),
            VLLM_DP_MASTER_IP=os.environ.get("NEMOTRON_DP_MASTER", "127.0.0.1"),
            VLLM_DP_MASTER_PORT=str(dp_port),
        )
    result = Path(os.environ["NEMOTRON_RESULT"])
    if dp_size > 1:
        result = result.with_name(f"{result.stem}-rank{dp_rank}{result.suffix}")
    if result.exists():
        raise FileExistsError(result)
    result.parent.mkdir(parents=True, exist_ok=True)
    token_budget = int(os.environ.get("NEMOTRON_TOKEN_BUDGET", "2048"))
    temperature = float(os.environ.get("NEMOTRON_TEMPERATURE", "0"))
    prompt_lengths = tuple(map(int, os.environ.get("NEMOTRON_PROMPT_LENGTHS", "37,127,128,257").split(",")))
    response_length = int(os.environ.get("NEMOTRON_RESPONSE_LENGTH", "136"))
    max_model_len = int(os.environ.get("NEMOTRON_MAX_MODEL_LEN", "1024"))
    assert len(prompt_lengths) == 4 and min(prompt_lengths) > 0
    assert response_length > 0 and max(prompt_lengths) + response_length < max_model_len
    print(
        {
            "token_budget": token_budget,
            "temperature": temperature,
            "request_seed": 42,
            "prompt_lengths": prompt_lengths,
            "response_length": response_length,
            "max_model_len": max_model_len,
        },
        flush=True,
    )
    llm = LLM(
        model=os.environ["NEMOTRON_MODEL"],
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=token_budget,
        gpu_memory_utilization=0.45,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        logprobs_mode="raw_logprobs",
        tensor_parallel_size=1,
        enable_expert_parallel=dp_size > 1,
        max_logprobs=1,
        seed=42,
        hf_overrides={"nemotron_shared_norms": True} if os.environ.get("NEMOTRON_SHARED_NORMS") == "1" else {},
    )
    rng = random.Random(42)
    prompts = [[rng.randrange(100, 50000) for _ in range(n)] for n in prompt_lengths]
    # Every rank executes the same call count, with different request ordering.
    prompts = prompts[dp_rank % 4 :] + prompts[: dp_rank % 4]
    params = SamplingParams(temperature=temperature, seed=42, max_tokens=response_length, ignore_eos=True, logprobs=0)
    try:
        outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        results = []
        for p, output in zip(prompts, outputs, strict=False):
            sampled = output.outputs[0]
            tokens = list(sampled.token_ids)
            assert len(tokens) == response_length, (len(tokens), response_length)
            score = llm.generate(
                [TokensPrompt(prompt_token_ids=p + tokens)],
                SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0),
                use_tqdm=False,
            )[0]
            alone = llm.generate([TokensPrompt(prompt_token_ids=p)], params, use_tqdm=False)[0].outputs[0]
            diffs = []
            for j, token in enumerate(tokens):
                a = sampled.logprobs[j][token].logprob
                b = score.prompt_logprobs[len(p) + j][token].logprob
                if a != b:
                    diffs.append({"position": j, "decode": a, "prefill": b})
            batch_equal = tokens == list(alone.token_ids) and all(
                sampled.logprobs[j][t].logprob == alone.logprobs[j][t].logprob for j, t in enumerate(tokens)
            )
            row = {
                "prompt_length": len(p),
                "prompt": p,
                "tokens": tokens,
                "rollout_logprobs": [sampled.logprobs[j][t].logprob for j, t in enumerate(tokens)],
                "prefill_mismatches": diffs,
                "batch_equal": batch_equal,
            }
            results.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k not in ("tokens", "prompt", "rollout_logprobs")}),
                flush=True,
            )
        result.write_text(json.dumps(results, indent=2))
        assert all(not r["prefill_mismatches"] and r["batch_equal"] for r in results)
        if os.environ.get("NEMOTRON_ANCHOR"):
            anchors = json.loads((result.parent / os.environ["NEMOTRON_ANCHOR"]).read_text())
            anchors = {tuple(row["prompt"]): row for row in anchors}
            for row in results:
                reference = anchors[tuple(row["prompt"])]
                assert row["tokens"] == reference["tokens"], ("anchor tokens", dp_rank)
                assert row["rollout_logprobs"] == reference["rollout_logprobs"], ("anchor logprobs", dp_rank)
            print("EP_VS_EP1_EXACT", dp_rank, flush=True)
        print("NEMOTRON7_INFERENCE_EXACT", flush=True)
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    size = int(os.environ.get("NEMOTRON_DP", "1"))
    if size == 1:
        main()
    else:
        import torch.multiprocessing as mp
        from vllm.utils.network_utils import get_open_port

        local_size = int(os.environ.get("NEMOTRON_DP_LOCAL", str(size)))
        assert size % local_size == 0
        if local_size != size:
            assert os.environ.get("NEMOTRON_DP_MASTER") and os.environ.get("NEMOTRON_DP_PORT")
        port = int(os.environ["NEMOTRON_DP_PORT"]) if os.environ.get("NEMOTRON_DP_PORT") else get_open_port()
        mp.spawn(main, args=(size, port), nprocs=local_size, join=True)
