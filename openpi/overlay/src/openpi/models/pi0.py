# Modified for the 2026 BEHAVIOR Challenge (team Dexterity) -- see openpi/CHANGES.md in the training repository.
import logging

import os

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


# B1K addition (opt-in: B1K_CORRELATED_NOISE=1 with B1K_NOISE_COV=<npz from
# an internal analysis script (compute_noise_cov.py)>): the 2025 winner's correlated flow-matching noise (arXiv 2512.06951 §5.2-5.4).
# Instead of isotropic N(0, I), draw eps = L z with L = chol(beta * Sigma + (1 - beta) I), Sigma the empirical
# covariance of normalised action chunks, so the noise already carries the joint-to-joint and step-to-step structure
# of real actions. It is used at EVERY draw -- both training loss paths and sample_actions -- because inference must
# start from the same distribution the velocity field was trained against. The authors report no ablation
# (§9.3, "did not have the budget"); our project notes measures it. Default off == stock, bit for bit.
_NOISE_L = None


def _noise_chol():
    global _NOISE_L
    # Cache the factor as a NUMPY array and convert per call. Caching a jnp array fails: the first call happens
    # while the training step is being traced, the cached value is then a tracer, and the next un-jitted call
    # (sample_actions in the end-of-run evaluation) raises UnexpectedTracerError (E5, C-s1/C-s2, 26 Sep).
    if _NOISE_L is None:
        import numpy as np
        _NOISE_L = np.asarray(np.load(os.environ["B1K_NOISE_COV"])["L"], dtype=np.float32)
    return jnp.asarray(_NOISE_L)


def _draw_noise(rng: at.KeyArrayLike, shape) -> at.Array:
    """Flow-matching noise of `shape` (..., horizon, action_dim): isotropic, or correlated when the switch is on."""
    z = jax.random.normal(rng, shape)
    if os.environ.get("B1K_CORRELATED_NOISE") != "1":
        return z
    L = _noise_chol()
    flat = z.reshape(*shape[:-2], shape[-2] * shape[-1])
    if flat.shape[-1] != L.shape[0]:
        raise ValueError(f"B1K_NOISE_COV factor is {L.shape[0]}-D but the model draws {flat.shape[-1]}-D noise")
    return (flat @ L.T).reshape(shape)          # eps = L z, written for row-vector batches


_STAGE_CLASSES = 15        # the largest N assigned by scripts/compute_stage_table.py (training repo)
_STAGE_LOSS_WEIGHT = 0.1   # the weight the 2025 winner used for their stage head


def _stage_div(action_horizon: int) -> int:
    """The per-step loss [b, ah] is AVERAGED over ah by the trainer, so a term broadcast onto every step already
    counts once. Dividing it by ah as well (the original code) made the stage weight 0.1/32 = 0.003, not the winner's
    0.1 (found 27 Sep, T64). B1K_STAGE_LOSS_FIX=1 applies the intended 0.1; unset keeps the tested legacy weight."""
    return 1 if os.environ.get("B1K_STAGE_LOSS_FIX") == "1" else action_horizon


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                # same switch as the Gemma stack above -- see gemma.py; default keeps upstream behaviour
                remat_policy=os.environ.get("B1K_REMAT_POLICY", "nothing_saveable"),
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # B1K addition (opt-in: B1K_STAGE_AUX=1): a linear head predicting the temporal stage of
        # the episode from the VLM's last prefix token. Auxiliary target only -- the prediction is never fed back,
        # so inference is unchanged. Created only when enabled, which keeps the default parameter tree identical
        # and therefore keeps checkpoints interchangeable. See our project notes.
        self.stage_head = (
            nnx.Linear(paligemma_config.width, _STAGE_CLASSES, rngs=rngs)
            if os.environ.get("B1K_STAGE_AUX") == "1"
            else None
        )
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # --- B1K addition: multi-sample flow matching, ported from the 2025 winning solution.
        # Default 1 == the stock single-sample loss, bit-for-bit. Set B1K_NUM_FLOW_SAMPLES to enable.
        # The env var (rather than a CLI flag) is the established pattern here because `base_config` is
        # tyro.conf.Suppress and cannot be overridden from the command line -- see our project notes.
        self.num_flow_samples = int(os.environ.get("B1K_NUM_FLOW_SAMPLES", "1"))

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.num_flow_samples > 1:
            return self._compute_loss_multi_sample(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = _draw_noise(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # Auxiliary stage loss (opt-in). Cross-entropy on the last prefix token, weight 0.1 -- the weight the 2025
        # winner used. Spread over the action-horizon axis so the returned shape is unchanged for every caller.
        if self.stage_head is not None and observation.stage is not None:
            logits = self.stage_head(prefix_out[:, -1])
            log_p = jax.nn.log_softmax(logits, axis=-1)
            stage_ce = -jnp.take_along_axis(log_p, observation.stage[:, None].astype(jnp.int32), axis=-1)[:, 0]
            loss = loss + _STAGE_LOSS_WEIGHT * stage_ce[:, None] / _stage_div(self.action_horizon)

        return loss

    def _compute_loss_multi_sample(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        """Average the flow-matching loss over N independent noise/time draws that SHARE one prefix forward pass.

        WHY THIS IS THE RIGHT LEVER FOR US. Our π0.5 training is input-bound, not compute-bound: measured 17.8
        samples/s from the loader against ~78 samples/s the GPUs can consume (phase-3, T9). The expensive part of a
        step is the prefix -- 3 images through SigLIP plus the VLM -- and that is exactly the part that depends on
        the sample we had to read from disk. The action expert is small by comparison.

        So: embed the prefix ONCE, cache its KV, then push N independent (noise, time) draws through the action
        expert against that cache. Each loaded sample yields N flow-matching targets instead of 1, which buys
        gradient signal per BYTE READ -- the quantity we are actually short of. It does not make the loader faster;
        it makes each loaded sample worth more, which is the same thing for wall-clock-to-convergence.

        Ported from the 2025 winner (`num_flow_samples=15` in their training config). Implemented with vmap over
        RNG keys, with the KV cache CLOSED OVER rather than tiled: tiling would materialise N copies of the prefix
        keys/values (~120 MB per batch element for our prefix length) and blow up memory, whereas a closed-over
        constant is broadcast.
        """
        preprocess_rng, flow_rng = jax.random.split(rng, 2)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        batch_size = actions.shape[0]

        # 1. prefix once -> KV cache. Gradients still flow into the prefix: the cache is a traced array, not a stop.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        def one_flow_sample(sample_rng):
            noise_rng, time_rng = jax.random.split(sample_rng, 2)
            noise = _draw_noise(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            # suffix queries attend to the whole prefix plus causally within the suffix -- identical construction to
            # sample_actions, so the attention pattern a sample sees in training matches the one it sees at eval.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            pre_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([pre_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

        flow_rngs = jax.random.split(flow_rng, self.num_flow_samples)
        with at.disable_typechecking():                       # jaxtyping cannot see through vmap's traced batch axis
            per_sample = jax.vmap(one_flow_sample)(flow_rngs)  # (N, *b, ah)
        loss = jnp.mean(per_sample, axis=0)
        # The stage-head loss, exactly as in the single-sample path. Without this, B1K_NUM_FLOW_SAMPLES>1 silently
        # dropped the auxiliary loss -- the head got no gradient and the feature was off while its switch said on
        # (found 25 Sep, the T46 class). The prefix is not attended to by the suffix in the other direction, so this
        # prefix-only pass gives the same prefix_out as the joint pass does.
        if self.stage_head is not None and observation.stage is not None:
            logits = self.stage_head(prefix_out[:, -1])
            log_p = jax.nn.log_softmax(logits, axis=-1)
            stage_ce = -jnp.take_along_axis(log_p, observation.stage[:, None].astype(jnp.int32), axis=-1)[:, 0]
            loss = loss + _STAGE_LOSS_WEIGHT * stage_ce[:, None] / _stage_div(self.action_horizon)
        return loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = _draw_noise(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
